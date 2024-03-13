import ollama
import os

class LLavaOllama():
    @staticmethod
    def run(image_path:str,prompt:str,model_name = "llava",stream=True):
        # Check variables
        # if not os.path.exists(image_path):
        #     raise Exception("Image Path is invalid")

        # Get the output
        output = ollama.generate(
            model=model_name,
            prompt=prompt,
            images=[image_path],
            stream=stream
        )
        return output