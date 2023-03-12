from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def InstructABSA(dataset):
    """
    https://github.com/kevinscaria/InstructABSA
    State of the art model
    Winning SemEval 2014 Task 4 Sub Task 2
    0.8829787234042553 accuracy for devdata.csv
    """
    tokenizer = AutoTokenizer.from_pretrained("kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined")
    model = AutoModelForSeq2SeqLM.from_pretrained("kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined")

    bos_instruct = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
        output: positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
        output: positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. The aspect is toast.
        output: negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches. The aspect is seats.
        output: negative
        Neutral example 1-
        input: I asked for seltzer with lime, no ice. The aspect is seltzer with lime.
        output: neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another. The aspect is glass of wine.
        output: neutral
        Now complete the following example-
        input: """
    delim_instruct = ' The aspect is '
    eos_instruct = '.\noutput:'
    outputs = []
    for i in range(len(dataset)):
        text = dataset['sentence'].iloc[i]
        aspect_term = dataset['target_term'].iloc[i]
        tokenized_text = tokenizer(bos_instruct + text + delim_instruct + aspect_term + eos_instruct,
                                   return_tensors="pt")
        output = model.generate(tokenized_text.input_ids)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(output)

    acc = sum(dataset['polarity'] == outputs) / len(outputs)
    print(f'The accuracy is {acc}')
    return outputs