import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json

image_folder = "SciVQA/images_validation/images_validation"

# Load the JSON file
with open("path/to/inference_log.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("validation_2025-03-27_18-34-44.json", "r", encoding="utf-8") as file:
    validation_data = json.load(file)

def wrap_text(text, max_width=50):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)

labeled_data = []
number_of_images = len(data)
# Extract the relevant question and response
for i, entry in enumerate(data):
    print(f"Entry {i + 1}/{number_of_images}")
    # Find the corresponding entry in validation_data with the same instance_id
    #validation_entry = next((v_entry for v_entry in validation_data if v_entry["instance_id"] == entry["instance_id"]), None)
    #if validation_entry:
    #    entry.update(validation_entry)

    image_file = entry["meta_data"]["image_path"]
    question = entry["meta_data"]["question"]
    reference_answer = entry["meta_data"]["reference_answer"]
    #image_file = entry["image_file"]
    #question = entry["question"]
    #reference_answer = entry["answer"]
    response = entry["response"].removeprefix("Answer: ")
    #if not "infinite" in entry["qa_pair_type"]:
    #    continue

    if response.lower() == reference_answer.lower():
        entry["answer_is_correct"] = True
        labeled_data.append(entry)
        continue
    if reference_answer.lower() in ["yes", "no"] and response.lower() in ["yes", "no"]:
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue
    if reference_answer.lower() in ["a", "b", "c", "d"]:
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue
    if reference_answer.lower() == "It is not possible to answer this question based only on the provided data.".lower():
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue

    answer_options = entry["context"]["prompt_vars"]["answer_options"]
    #answer_options = entry["answer_options"]
    image_path = os.path.join(image_folder, image_file)
    
    # Load and display the image
    img = mpimg.imread(image_path)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')  # Hide axes

    # Add the question and response as text
    # Function to add line breaks for long text
    def wrap_text(text, max_width=100):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_width:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    # Wrap the question and response text
    wrapped_question = wrap_text(question)
    plt.title(wrapped_question)

    wrapped_response = wrap_text(response)

    # Adjust vertical spacing to ensure all text is visible
    #plt.text(0.5, -0.1, f"Question: {wrapped_question}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    # Wrap the reference answer text
    wrapped_reference_answer = wrap_text(reference_answer)

    # Adjust vertical spacing dynamically based on the length of the wrapped response
    response_y_position = -0.1 - 0.05 * wrapped_response.count('\n')
    reference_answer_y_position = response_y_position - 0.1 - 0.05 * wrapped_reference_answer.count('\n')

    plt.text(0.5, response_y_position, f"Response: {wrapped_response}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, reference_answer_y_position, f"Reference Answer: {wrapped_reference_answer}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    print("-------------------------------------------------------------------------------")
    print(f"Reference Answer: {reference_answer}")
    print(f"Response: {response}")
    #print(entry["raw_response"])

    # Show the plot
    plt.tight_layout()
    def on_key(event):
        if event.key == 'left':
            entry["answer_is_correct"] = False
            labeled_data.append(entry)
            plt.close()
        elif event.key == 'right':
            entry["answer_is_correct"] = True
            labeled_data.append(entry)
            plt.close()

    # Connect the key press event to the handler
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    # Save the modified data to a new JSON file after all entries are processed
    with open("inference_log_32B_2Epochen_Val_Annotated.json", "w", encoding="utf-8") as output_file:
        json.dump(labeled_data, output_file, ensure_ascii=False, indent=4)

# Save the modified data to a new JSON file after all entries are processed
with open("inference_log_32B_2Epochen_Val_Annotated.json", "w", encoding="utf-8") as output_file:
    json.dump(labeled_data, output_file, ensure_ascii=False, indent=4)