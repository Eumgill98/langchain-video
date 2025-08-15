# ImageBind: One Embedding Space To Bind Them All

**[FAIR, Meta AI](https://ai.facebook.com/research/)** 

Rohit Girdhar*,
Alaaeldin El-Nouby*,
Zhuang Liu,
Mannat Singh,
Kalyan Vasudev Alwala,
Armand Joulin,
Ishan Misra*

To appear at CVPR 2023 (*Highlighted paper*)

For details, see the paper: **[ImageBind: One Embedding Space To Bind Them All](https://facebookresearch.github.io/ImageBind/paper)**.


## API Usage

Install pytorch 1.13+ and other 3rd party dependencies. (check `requirements.txt`)

Extract and compare features across modalities (e.g. Image, Text and Audio).

```python
from lagnchain_video.embeddings.imagebind.imagebind import ImageBindEmbeddings

# initiate embedding model
embedding_model = ImageBindEmbeddings(device="cuda") # or "cpu"

# single text
text_embedding = embedding_model.embed_query_text("A dog.") # text should be string

# multiple texts
text_embeddings = embedding_model.embed_documents(["A dog.", "A cat."]) # list of texts

# single image
image_embedding = embedding_model.embed_query_image(image) # image can be file_path, np.ndarray, etc.

# multiple image
image_embeddings = embedding_model.embed_images(images) # list of images

# single audio
# audio input must be a tuple: (file_path or np.ndarray or etc, sampling_rate or None)
audio_embedding = embedding_model.embed_query_audio(audio)

# multiple audio
audio_embeddings = embedding_model.embed_audios(audios) # list of audios

```

## License

Original ImageBind code and model weights are released under the CC-BY-NC 4.0 license. So this code are also **released under the CC-BY-NC 4.0 license.**

## Reference

```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```