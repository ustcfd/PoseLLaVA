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
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
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
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        # if len(state.get_images(return_pil=True)) > 0:
        #     state = default_conversation.copy()
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", text)
    # if len(state.get_images(return_pil=True)) > 0:
    #     state = default_conversation.copy()

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False

    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


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
<h1 align="center"> PoseLlava: Pose Centric Multimodal LLM for Fine-Grained 3D Pose Manipulation </h2>
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
                    [
                        f"{cur_dir}/examples/S8_WalkTogether_1.58860488_000001.jpg", 
                        "There is a person in the middle of the image, use SMPL to describe the pose."
                    ],
                    [
                        f"{cur_dir}/examples/135000_A.png", 
                        # f"{cur_dir}/examples/135000_B.png", 
                        "The following description requires your attention. Raise your right leg and bend your left arm. Given the initial pose of a human in the image, can you predict and adjust the SMPL pose according to the textual description?"
                    ],
                    # [
                    #     f"{cur_dir}/examples/19247_A.png", 
                    #     "Kindly review the provided description. Sit on the floor with both of your legs crossed, left leg in front of your right leg. Move your arms out to the side slightly. Given the initial pose of a human in the image, can you predict and adjust the SMPL pose according to the textual description?"
                    # ],
                    # [
                    #     f"{cur_dir}/examples/135086.png", 
                    #     "Kindly review the provided description. Unbend completely the left arm then raise the left hand a little, move the left hand slightly leftwards, unbend completely the right arm then lift the right hand up a little then bring the right hand backward slightly then move the left foot forward slightly. Given the initial pose of a human in the image, can you predict and adjust the SMPL pose according to the textual description?"
                    # ],
                    # [
                    #     f"{cur_dir}/examples/135096.png", 
                    #     "The following description requires your attention. Move your left knee slightly rightwards while moving your left foot slightly rightwards then bring your left foot forward a little and your left foot must be vertically in line with your right hip and bring your right knee up slightly and bring your right foot up slightly while moving your right foot to the right. Commencing with the pose visible in the image, adapt it according to the textual guidance to produce a modified SMPL pose."
                    # ],
                    [   
                        None,
                        "What's the SMPL pose of this person? left leg is angled out to the side, head is slightly looking down to the left, elbows are bent back."
                    ],

                ], 
                inputs=[imagebox, textbox]
            )

            gr.Examples(
                examples=[
                    # [
                    #     f"{cur_dir}/examples/19242_A.png", 
                    #     f"{cur_dir}/examples/19242_B.png", 
                    #     "To achieve consistency between the human poses in pictures A and B, what modifications should be implemented?"
                    # ],
                    # [
                    #     f"{cur_dir}/examples/19243_A.png", 
                    #     f"{cur_dir}/examples/19243_B.png", 
                    #     "When transitioning from picture A to picture B, what adjustments should be applied to the human pose in A to match that of B?"
                    # ],         
                    # text2pose_13 and text2pose_78
                    [
                        f"{cur_dir}/examples/S8_Walking_1.60457274_000621.jpg", 
                        "What is the human pose in this image? Please respond with SMPL pose."
                    ],
                    [
                        # f"{cur_dir}/examples/135001_A.png", 
                        f"{cur_dir}/examples/135001_A.png", 
                        "Read the following text description: let your arms hang down naturally, keep your head facing forward. Your goal is to synthesize a SMPL pose based on the human pose depicted in the image, refining it based on the details provided in the text."
                    ],
                    # [
                    #     # f"{cur_dir}/examples/135001_A.png", 
                    #     f"{cur_dir}/examples/19251_A.png", 
                    #     "Take a moment to read the following passage. Lean over slightly and look slightly down. Extend your arms straight out in front of you slightly raised. Put your left leg out in front of you raised. Given the initial pose in the image and the textual description, try adjusting the SMPL pose to match the requirements outlined in the description."
                    # ],
                    [
                        None,
                        "There is a person doing this: left arm is straight up, right arm is straight out to the side, left leg is forward, head is forward. Can you use SMPL pose to describe the pose?"
                    ]
                ], 
                inputs=[imagebox, textbox]
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
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
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