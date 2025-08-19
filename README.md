# 🎬 langchain-video
<img src="./resource/logo.png" alt="logo" width="200" >

**`langchain-video`** is a LangChain-based library for **multimodal data processing**, enabling video, image, and audio embedding, storage, and retrieval. Seamlessly integrate with LangChain RAG pipelines or other LLM workflows.  

---
## 🚀 Key Features

### 🎨 Multimodal Data Support
- Handle **images, videos, and audio** in a unified API
- Preprocessing and embedding support
- Single or batch data processing

### 📹 Video Processing
- Split videos into **clips** and embed each segment
- `FFmpeg`-powered video loading and transformations

### 💾 Vector Store Integration
- Supports `MultiModalVectorStore` and `MultiModalFAISS`
- Add and search embeddings for images, audio, and video
- Fully compatible with LangChain VectorStore interface

### 🤖 LangChain Integration
- Plug directly into **RAG (Retrieval-Augmented Generation)** pipelines
- Retrieve video content and connect it to LLMs for Q&A

---
## ✅ Installation
### 1. Install `FFmpeg` **(Prerequisites)**
- `Ubuntu/Debian`:
    ```
    sudo apt update
    sudo apt install ffmpeg
    ```
- `macOS (using Homebrew)`:
    ```
    brew install ffmpeg
    ```

- `Windows`:
    - Download from [FFmpeg official website](https://ffmpeg.org/download.html)
    - Extract and add to `PATH`
    - Or use Chocolatey: choco install ffmpeg

### 2. Optional Package
`Pytorch (cuda)`
- if you want to use `cuda`,
- Choose the appropriate CUDA version for your GPU:  
```bash
# Example: PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
---
`FAISS (GPU)`
```
pip install faiss-gpu-{your_cuda_version}

# Example "pip install faiss-gpu-cu12`
```

### 3. Install `langchain-video`
```bash
# Using HTTPS
pip install git+https://github.com/Eumgill98/langchain-video.git

# Or clone and install locally
git clone https://github.com/Eumgill98/langchain-video.git
cd langchain-video
pip install .

```
---

## 🛠️ TODO

- [ ] Add pipelines specialized for various **video tasks**
- [ ] Support additional **multimodal embedding models**
- [ ] Add support for more **multimodal vector stores**
- [ ] Integrate with **LangSmith** for workflow management