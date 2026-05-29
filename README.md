# Reading the Rear
## Overview
For our LLM identification pipeline, we take in pre-processed images of rearview of vehicles. This system focused on identifying the visible sticker and decals on the vehicle, which are then treated as quasi-identifiers, and are fed and processed into the Claude LLM. Based on these inputs, the model outputs inferred attributes about the driver of the car, including their beliefs, affiliations, and lifestyle.  
Although these quasi-identifiers given to the LLM system may seem inconsequential by themselves, with the collection of multiple points of data from the vehicle, it creates a more vivid picture and identification of the owner of the vehicle. Demonstrating how with just publicly observable data, if used in conjunction with LLMs, it can be used to create profiles on the owners of these vehicles, raising many red flags on privacy. 
## Assumptions on the Adversary
In our testing, the adversary is assumed to be a non-privileged actor with:
- Access to publicly available and visible information like camera captures of roads, parking lots, etc.
- Access to a general purpose LLM
- In our Model we use Claude Haiku 4.5 (claude-haiku-4-5-20251001)
## Adversary Goals
The goal of this potential adversary is to:
- Infer and/or deduct sensitive information about a vehicle’s owner, including:
- Their political or social beliefs
- Affiliations (Their place of work, or school)
- General locations of their job, work, or third place
- Narrow down the demographic of the vehicle owner 
## Available Information
An attacker would have access to the following information:
- Rear view images of vehicles, including their:
- Stickers and decals on the rear view (primary quasi identifier)
- The brand and type of vehicle
- Other potential information sources:
- The location and time the photo was taken
- License plate number
- Note: In our model the attacker does not have access to a DMV database in our model
## Assumptions
In our threat model, the following assumptions are made when an attack is carried out:
- The attacker is not interested in the information given from license plates
- Stickers are signposts that reflect the identity and beliefs of the owner of the vehicle
- LLMs follow the most common patterns and associations that they were trained on
- The outputs are not guaranteed to be correct
## Threat Pipeline
### Steps:
1. Capture rear image of target vehicle that includes the quasi-identifiers noted before. The higher quality of image or video the better. Captured over 40 GB of raw footage. Footage was legally captured in public from a short distance with a HERO 5 GoPro. Roughly analogous to last-generation Tesla car-mounted cameras. 
2. Using a YOLOv10 nano model, we crop the photo of extraneous picture data so it only contains a crop of the car. During this step we also apply a gaussian blur to the license plate. Processing produced over 7,000  images.
3. With the cropped image, we give it as input to the LLM as well as the prompt shown in 3.2.5.2.
4. The LLM returns a formatted JSON file that includes the inferred location of each sticker, a label from a pre-defined list, and its reasoning for putting that label
## Limitations
- Of all of the rears of cars captured, **many did not have stickers**, limiting the size of our dataset.
- Of all of our cars captured, they are all taken at around the same area, limiting the diversity of our dataset
- Our output is based on LLM inferences, not confirmed identification
- Some images and videos were of lower quality than desired when collecting data
- There is quite a bit of model hallucination in our output
- Our choice of a smaller model, Claude Haiku, can lead to lower quality results
- Some stickers do not fall under just one category. Our current LLM model does not show the nuances of some more complex stickers
