import streamlit as st
import time
from components.ai_modules.itt_modules.ollama_llava import LLavaOllama
from components.system_module.gpu_modules import GpuModule

# Define name
class_name = LLavaOllama.__class__.__name__

# @st.cache_resource
def stream_answer(prompt,image_path):
    # Get answer
    output = LLavaOllama.run(image_path=image_path, prompt=prompt)
    # Stream
    for part in output:
        yield part['response']

def gpu_status():
    # Display status
    _, total_vram, used_vram, usage = GpuModule.get_gpu_property()
    _, total_vram, used_vram = GpuModule.formmated_property(_, total_vram, used_vram)
    return  total_vram, used_vram, usage

def generate_output(prompt,image_path):
    st.write_stream(stream_answer(prompt=prompt,image_path=image_path))

def main():
    # Create History sidebar
    with st.sidebar:
        total_vram, used_vram, usage = gpu_status()
        st.title("Configuration")
        st.subheader("% VRAM Usage")
        st.progress(usage)
        st.caption(f"{used_vram}/{total_vram}GB")

    # Main menu
    # Header
    st.title(f"Demo Image2Text model with {class_name}")
    # Create space
    st.write("##")
    # Upload file field
    st.header("Image:")
    uploaded_file = st.file_uploader("Upload file",type=['png','jpg'],label_visibility="hidden")

    # Check image file is None
    if uploaded_file is not None:
        byte_file = uploaded_file.getvalue()
        # Display image file
        st.image(byte_file)

        # Create space
        st.write("##")
        st.write("##")
        # Prompt field
        st.header("Prompt:")
        prompt = st.text_input("Prompt",placeholder="Describe this picture",label_visibility="hidden")

        # Apply button
        clicked_button = st.button("Apply")
        if clicked_button:
            # Begin time
            beginTime = time.time()
            # Stream output
            generate_output(prompt=prompt,image_path=byte_file)
            # Output time
            endTime = time.time() - beginTime
            # Show status
            st.caption(f"Time consuming: {round(endTime,2)}s")
            st.toast(f"Done! {round(endTime,2)}s",icon="😍")

if __name__ == "__main__":
    main()