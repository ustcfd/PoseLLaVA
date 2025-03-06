import argparse
import datetime
import json
import os
import time
import torch
import gradio as gr
import requests

from mllm.utils.log_utils import (LOGDIR, build_logger, server_error_msg,
                                violates_moderation, moderation_msg)
from mllm.conversation import get_conv_template, SeparatorStyle, save_single_image

import hashlib
from PIL import Image
from .posechat_worker import ModelWorker

from loaders.model_loaders import LOADERS


logger = build_logger("gradio_web_server_local", "gradio_web_server_local.log")

headers = {"User-Agent": "LLaVA Client"}
# global template_name 

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

def replace_pose_token(original_string):
    modified_string = original_string.replace("<POSE>", "<span style='color:red;font-weight: bold;'>`<POSE>`</span>")
    return modified_string

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    state = get_conv_template(template_name).copy()
    return state


def vote_last_response(state, vote_type, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = get_conv_template(template_name).copy()
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def add_text(state, text, image, image_2, image_process_mode, request: gr.Request):
    # logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None and image_2 is None:
        text = text[:960]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
    elif image is None and image_2 is not None:
        text = text[:960]
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image_2, image_process_mode)
      
    elif image is not None and image_2 is not None:
        text = text[:500]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>' + '\n<image>'
        text = (text, [image, image_2], image_process_mode)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False

    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def http_bot(state, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    # start_tstamp = time.time()
    # if state.skip_next:
    #     # This generate call is skipped due to invalid inputs
    #     yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
    #     return
    if len(state.messages) == state.offset + 2:
        new_state = get_conv_template(template_name).copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)
    
    # Make requests
    pload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        # response = requests.post(worker_addr + "/worker_generate_stream",
        #     headers=headers, json=pload, stream=True, timeout=10)
        # for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        response = model.generate_stream_gate(pload)
        pose_img_path = None
        for chunk in response:
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = replace_pose_token(output) + "▌"   
                    # state.messages[-1][-1] = output + "▌"           
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                elif data["error_code"] == 1:               
                    pose_img_path = data["text"].strip()
                    # pose_data = Image.open(pose_img_path).convert('RGB')
                    # state.messages[-1][-1] = pose_img_path + "▌"           
                    # yield (state, state.to_gradio_chatbot(), None) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = replace_pose_token(output)
                    # state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return
    
  
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    if pose_img_path is not None:
        pose_img = Image.open(pose_img_path).convert('RGB')
        pose_img_b64_str = save_single_image(pose_img)
        pose_img_str = f'<img src="data:image/png;base64,{pose_img_b64_str}"/>'
        final_pose_img_str = "<span style='color:blue; font-weight: bold;'>pose visualization:</span>" + pose_img_str
        
        state.append_message("Pose", None)
        state.append_message("Pose", final_pose_img_str)
    
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    # finish_tstamp = time.time()
    # logger.info(f"{output}")

    # with open(get_conv_log_filename(), "a") as fout:
    #     data = {
    #         "tstamp": round(finish_tstamp, 4),
    #         "type": "chat",
    #         "start": round(start_tstamp, 4),
    #         "finish": round(start_tstamp, 4),
    #         "state": state.dict(),
    #         "images": all_image_hash,
    #         "ip": request.client.host,
    #     }
    #     fout.write(json.dumps(data) + "\n")





title_markdown = ("""
<h1 align="center"><a href="https://github.com/X-PLUG/mPLUG-Owl"><img src="https://z1.ax1x.com/2023/11/03/piM1rGQ.md.png", alt="mPLUG-Owl" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>

<h2 align="center"> PoseLlava: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation </h2>

<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/X-PLUG/mPLUG-Owl'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href="https://arxiv.org/abs/2304.14178"><img src="https://img.shields.io/badge/Arxiv-2304.14178-red"></a>
        <a href='https://github.com/X-PLUG/mPLUG-Owl/stargazers'><img src='https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social'></a>
    </div>
</div>

""")


tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode):
    
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="posellava", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")
                imagebox_2 = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))


                with gr.Accordion("Parameters", open=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=10):
                chatbot = gr.Chatbot(elem_id="Chatbot", label="LLaVA Chatbot", height=600)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
            
        with gr.Row():
        
             gr.Examples(
                examples=[
                    # [
                    #     f"{cur_dir}/examples/6573_A.png", 
                    #     f"{cur_dir}/examples/6573_p4_C.png", 
                    #     "Examine two sequential pose images and determine which body part has changed the most from the options: left upper limb, right upper limb, left lower limb, right lower limb, torso, and head. How should the pose from the first image be adjusted to align with the second image? Please describe the necessary changes."
                    # ],
                    [
                        f"{cur_dir}/examples/6578_A.png", 
                        f"{cur_dir}/examples/6578_p2_C.png", 
                        "Examine two sequential pose images and determine which body part has changed the most from the options: left upper limb, right upper limb, left lower limb, right lower limb, torso, and head. How should the pose from the first image be adjusted to align with the second image? Please describe the necessary changes."
                    ],
                    [
                        f"{cur_dir}/examples/6572_A.png", 
                        f"{cur_dir}/examples/6572_p1_C.png", 
                        "Review the two pose images in sequence and determine the body part that has changed the most among these options: left upper limb, right upper limb, left lower limb, right lower limb, torso, and head. What modifications should be made to align the first pose with the second? Please elaborate."
                    ],
                    
                ], 
                inputs=[imagebox, imagebox_2, textbox]
            )
        
        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            state,
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            state,
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            state,
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox, imagebox_2] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox, imagebox_2] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, imagebox_2, image_process_mode],
            [state, chatbot, textbox, imagebox, imagebox_2] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, imagebox_2, image_process_mode],
            [state, chatbot, textbox, imagebox, imagebox_2] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        demo.load(
            load_demo,
            [url_params],
            state,
            _js=get_window_url_params,
            queue=False
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model_family_id", type=str, default="llava-13b")
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv_style", type=str, default="llava_v1")

    parser.add_argument("--concurrency_count", type=int, default=10)
    parser.add_argument("--model_list_mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--moderate", action="store_true")
 
    
    args = parser.parse_args()
    global template_name
    template_name = args.conv_style
    logger.info(f"args: {args}")
    model = ModelWorker(model_family_id =args.model_family_id, 
                        model_name_or_path = args.model_path, 
                        device= args.device)

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )