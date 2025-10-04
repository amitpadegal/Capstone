from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from vqa import blip_vqa
import os
import json
import random

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
        transforms.Resize((480, 480),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
# import io
# print(os.path.exists("image/dog_image.jpeg"))
# with open("image/dog_image.jpeg", "rb") as f:
#     image = Image.open(io.BytesIO(f.read())).convert("RGB")
def jumble_sentence(sentence: str) -> str:
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)

with open("/Users/amitpadegal/Desktop/Capstone/vqa_20k_subset.json", "r") as f:
    data = json.load(f)
# print(data[:1000])  # Print the first 1000 characters to verify content
questions = [
    "Can you describe the color of the phone that the woman is holding in her hand?",
    "Is the cat in the picture currently wearing a seat belt while sitting down?",
    "Would a woman generally feel comfortable if she had to use this particular bathroom?",
    "Is the woman in the scene wearing an apron over her clothes at the moment?",
    "Can you tell me what exactly is written or displayed on the sign in the picture?",
    "Is the zebra in the image currently standing upright or is it lying down on the ground?",
    "Do you know the destination to which the bus in the image is going?",
    "What specific object or tool are they using to row the boat in the picture?",
    "What type or kind of car is visible on the right side of the image?",
    "Which particular kitchen utensil is the woman using while she is working?",
    "Are there any dirty dishes present inside the sink in this picture?",
    "What brand or company name of bicycle can be seen in the image?",
    "Does the weather in the picture appear to be cloudy or overcast?",
    "At what place or location can you see the giraffes in this image?",
    "Would you say that the room in this picture appears to be clean and tidy?",
    "Can you tell me what numbers are displayed on the front of the train in this image?",
    "Could you describe how many seating areas or sections are visible in this picture?",
    "Is the man shown in the image paying close attention or being attentive to something?",
    "Which of the chefs in the image is wearing a black apron over their clothing?",
    "Looking at the pizzas in the picture, which one has more slices remaining or left uneaten?",
    "Can you identify whether the computer shown in the image is a desktop model?",
    "Is there anyone currently inside the shower that can be seen in this picture?",
    "Are the zebras shown in the image all facing the same direction or orientation?",
    "Are there people present within the scene, and can you tell how many?",
    "How many donuts are positioned next to each other on the surface in the image?",
    "Can you determine who is taking a picture in this scene or holding the camera?",
    "How many towers can be seen in the background of this image?",
    "Can you identify which city or urban area is shown in the picture?",
    "Who is the man wearing the black shirt in this image, if he can be identified?",
    "Can you describe how the computer in the image is currently turned on or functioning?",
    "Can you tell me how many lights in the picture are currently turned on or lit?",
    "Does the man shown in the image have any facial hair such as a beard or mustache?",
    "How many chairs have been arranged or placed around this table in the scene?",
    "Could you describe the letters or markings that are visible on the side of the snowmobile?",
    "How many different colors of briefcases can be seen in this image?",
    "Can you identify the specific color or shade of the floor shown in the picture?",
    "What type of food is the zebra currently eating in the image?",
    "Are portions of the chicken in the picture covered with a glazed sauce or coating?",
    "Can you determine whether the animal shown is a black bear?",
    "Does the man in the image have any hair visible on his chest area?",
    "Is the kite shown in the image flying high up in the sky?",
    "Can you read and explain what is written on the plane in the image?",
    "Does the cell phone visible in the picture appear to be plain or decorated in any way?",
    "What is the color of the shirt worn by the woman in the forefront of the image?",
    "Based on the appearance and features, which room of the house does this look like?",
    "Can you tell me whether the bench is situated in the middle of a lake or somewhere else?",
    "Do the two giraffes shown in the image live or reside in the African grasslands?",
    "In which direction is the man wearing the black coat moving while on the escalator?",
    "Are there any trees visible around the scene in this image?",
    "Can you determine whether there is any meat present in the wok in this picture?",
    "Which room of the house does this image appear to show based on its contents and layout?",
    "Where are the orange traffic cones located within the scene captured in the image?",
    "Could you describe the locations of the benches that are visible in the image?",
    "What is unusual or distinctive about the clock shown in this picture?",
    "Based on the visual features, what room in the house does this appear to be?",
    "Can you identify the color of the benches that are shown in the image?",
    "Are the animals in the image positioned on the side of a mountain or elsewhere?",
    "How many people can be seen inside this train station according to the picture?",
    "What object or scene is reflected in the stove in this image?",
    "Is there a man in the image who is wearing a hat, and if so, which one?",
    "Can you tell me where the donuts in the image came from or which store they are from?",
    "Could you describe what object or item the woman is holding in her hands?",
    "What is the name of the store or shop that is located behind the parking meter in the picture?",
    "Is the cat in the image able to use or operate the remote control?",
    "Can this cupcake shown in the picture be eaten or is it for display only?",
    "Is the landing gear of the aircraft shown in the image currently down or extended?",
    "What is the animal in the image wearing on its body or as clothing?",
    "Are both hands of the person in the image holding an object or something else?",
    "Can you determine whether the people in the picture are American or not?",
    "Does the child in the image appear messy or disheveled?",
    "How many paper towels are left on the roll visible in the image?",
    "What type of room does this image depict based on the objects and layout?",
    "What is the name or function of the device that has a battery on its parts?",
    "Can you describe the color of the traffic light shown in this picture?",
    "What object is the man holding in his hands in the image?",
    "Can you determine whether the artwork shown in the image is actually a painting or something else?",
    "Is it currently raining in the scene captured by this picture, indicating active rainfall?",
    "Is the man standing on the far left dressed appropriately for the weather conditions shown in the image?",
    "Can you tell me the location of the passport in the scene?",
    "Is the person visible in the image wearing any kind of hat or head covering?",
    "What object or item is being used as a vase to hold flowers in this picture?",
    "How many people or persons are present in this particular scene?",
    "Can you identify the types or species of flowers that are placed on the table?",
    "How many large rocks can be seen in the image?",
    "How many boats are present within the frame of the photograph?",
    "Why do the zebras appear dirty in this image?",
    "Do the dogs shown in the picture know how to swim or are they just near water?",
    "What is the color of the car that is farthest from the photographer in this image?",
    "What is the color of the shirt worn by the man standing behind the bar in the picture?",
    "Can you determine the location or place where this photo was taken?",
    "Based on the image, do you think the surfer will wipe out while riding the wave?",
    "What is the name of the creature that is painted on the boat shown in the picture?",
    "What types of animals are the stuffed animals visible in this image?",
    "How many red donuts can be seen on the table in this picture?",
    "Can you describe the color of the couch that appears in the image?",
    "Are the zebras in the scene looking for someone or something specific?",
    "Have you ever experienced skiing in a manner similar to what is depicted here?",
    "What colors are the boats that are visible in this image?",
    "Is the sun shining in the scene captured by this picture?",
    "Is the motorcycle shown in the image for sale?",
    "Is the bed in the picture currently made or untidy?"
]
l = []
jumbled_sentences = [jumble_sentence(data[i]['question']) for i in range(15)]
model = blip_vqa(pretrained='model_vqa.pth')
for i in range(2):
        image_path = os.path.join('/Users/amitpadegal/Desktop/Capstone/vqa_sample', data[i]['image'].split('/')[1])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        original = data[i]['question']
        long = questions[i]
        jumble = jumbled_sentences[i]
        _, res_o = model(image, original, inference= 'generate')
        _, res_l = model(image, long, inference= 'generate')
        _, res_j = model(image, jumble, inference= 'generate')
        d = {}
        d['original'] = res_o
        d['long'] = res_l
        d['jumble'] = res_j
        l.append(d)
        # print(original, long, jumble)

print(l)
with open("blip_vqa_output_5.json", "w") as f:  
        json.dump(l, f, indent=2)
        
        # print(f"Q: {question}")
        # print(f"A: {logits}\n")

# image = Image.open("/Users/amitpadegal/Desktop/Capstone/BLIP-Analysis/image/dog_image.jpeg").convert('RGB')
# image = transform(image)
# image = image.unsqueeze(0)
# # print(image.shape)``
# question = "Given the distinct physical characteristics presented in this photograph, including the dog's coat texture, facial structure, and overall build, could you provide a precise identification of the canine breed depicted?"

# model = blip_vqa(pretrained='model_vqa.pth')

# logits = model(image, question, inference= 'generate')
# print(logits)
