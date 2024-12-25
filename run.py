from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from vqa import blip_vqa

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
        transforms.Resize((480, 480),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  

image = Image.open("images/dog_image.jpeg")
image = transform(image)
image = image.unsqueeze(0)
# print(image.shape)
question = "What is the colour of the clown's nose?"

model = blip_vqa(pretrained='model_vqa.pth')

logits = model(image, question, inference= 'generate')
print(logits)
