import dataclasses
from enum import auto, IntEnum
from typing import Any, Dict, List, Tuple, Union
from mllm.models.constants.llava_constants import DEFAULT_IMAGE_TOKEN
from mllm.utils.mm_utils import expand2square

import base64
from io import BytesIO

class SeparatorStyle(IntEnum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    TINY_LLAMA = auto()
    QWEN_2 = auto()


def handle_single_image(image, image_process_mode):
    if image_process_mode == "Pad":
        image = expand2square(image,background_color=(122, 116, 104))
    elif image_process_mode in ["Default", "Crop"]:
        pass
    elif image_process_mode == "Resize":
        image = image.resize((336, 336))
    else:
        raise ValueError(
            f"Invalid image_process_mode: {image_process_mode}")
    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 800, 400
    shortest_edge = int(
        min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if longest_edge != max(image.size):
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    return image


def save_single_image(image):
    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 800, 400
    shortest_edge = int(
        min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if H > W:
        H, W = longest_edge, shortest_edge
    else:
        H, W = shortest_edge, longest_edge
    image = image.resize((W, H))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(
        buffered.getvalue()).decode()
    return img_b64_str

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]]=()
    # The number of few shot examples
    offset: int =0
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    
    image_token : str = DEFAULT_IMAGE_TOKEN # Add one image_token
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self):
        system_prompt = self.system_template.format(system_message=self.system_message)
        
        # if len(messages) > 0 and type(messages[0][1]) is tuple:
        #     messages = self.messages.copy()
        #     init_role, init_msg = messages[0].copy()
        #     image_number = len(init_msg[1]) if isinstance(init_msg[1],list) else 1
        #     init_msg = init_msg[0].replace(self.image_token, "").strip() # 修改
        #     if 'mmtag' in self.version:
        #         messages[0] = (init_role, init_msg)
        #         messages.insert(0, (self.roles[0], f"<Image><{self.image_token}></Image>"*image_number))
        #         messages.insert(1, (self.roles[1], "Received."))
        #     else:
        #         messages[0] = (init_role, f"{self.image_token}\n"*image_number + init_msg)


        if self.sep_style == SeparatorStyle.SINGLE:
            # ret = self.system + self.sep
            ret = system_prompt + "\n\n" + self.sep + " "
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    # ret += role + ": " + message + self.sep
                    ret += role + ": " + message + "\n" + self.sep + " "
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.TINY_LLAMA:
            sep = "</s>"
            wrap_sys = lambda msg: f"<|system|>\n{msg}\n"
            wrap_user = lambda msg: f"<|user|>\n{msg}\n"
            wrap_assistant = lambda msg: f"<|assistant|>\n{msg}"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        message = wrap_user(message)
                        if i == 0:
                            message = wrap_sys(self.system) + message
                        ret += self.sep + message
                    else:
                        message = wrap_assistant(message) + self.sep2
                        ret += message
                else:
                    ret += "<|assistant|>\n"
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.QWEN_2:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        
        return ret
    
    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role, message):
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if isinstance(image, list):
                        image = [handle_single_image(img, image_process_mode) for img in image ]
                    else:
                        image = [handle_single_image(image, image_process_mode)]

                    if return_pil:
                        images.extend(image)
                    else:
                        for img in image:
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                            images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if isinstance(image, list):
                        img_str = f'<div style="display:flex;">'
                        for img in image:
                            single_img_b64_str = save_single_image(img)
                            img_str += f'<img src="data:image/png;base64,{single_img_b64_str}" alt="user upload image" style="margin-right: 10px;"/>'
                        img_str+='</div>'
                    else:
                        img_b64_str = save_single_image(image)
                        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'  
                    msg = img_str + msg.replace(self.image_token, '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                'template_name': self.name,
                'system_message': self.system_message,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }



# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name='llava_v1',
        system_message="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}',
        system_message="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    )
)







