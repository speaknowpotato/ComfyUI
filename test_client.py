from main import execution


class FakeServer:
    def __init__(self):
        self.client_id = None


class Pipeline:
    def __init__(self):
        self.executor = execution.PromptExecutor(FakeServer())
        self.prompt_id = ""
        self.number = 0

    def run_prompt(self, prompt):
        valid = execution.validate_prompt(prompt)

        print(f"valid: {valid}")
        extra_data = {}
        if valid[0]:
            outputs_to_execute = valid[2]
            self.executor.execute(
                prompt, self.prompt_id, extra_data, outputs_to_execute
            )
            print(f"outputs_to_execute: {outputs_to_execute}")
            print(f"history_result: {self.executor.history_result}")
            return self.executor.history_result
        else:
            print(f"Invalid prompt: {valid[1]}")

    def get_current_queue(self):
        return self.executor.get_current_queue()


if __name__ == "__main__":
    prompt = {
        "3": {
            "inputs": {
                "seed": 865054285191429,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "4": {
            "inputs": {"ckpt_name": "SD1.5/v1-5-pruned-emaonly.ckpt"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
        "5": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        "6": {
            "inputs": {"text": "a dog", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "7": {
            "inputs": {"text": "text, watermark", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        # "9": {
        #     "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
        #     "class_type": "SaveImage",
        #     "_meta": {"title": "Save Image"},
        # }
        "9": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"},
        }
        
        ,
    }
    pipeline = Pipeline()
    pipeline.run_prompt(prompt)
