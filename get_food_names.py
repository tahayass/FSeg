import json

# Open and read the JSON file
with open('./Combined_dataset/test/_annotations.coco.json', 'r') as file:
    data = json.load(file)

# Extract the names from the categories list
unique_names = set(category["name"] for category in data["categories"])


# Write all names to a new file
#with open('food_names.txt', 'w') as outfile:
#    for name in unique_names:
#        outfile.write(name + '\n')

# Write all names to a new file
with open('food_names_test.txt', 'w') as outfile:
    for name in unique_names:
        outfile.write(name + '\n')
