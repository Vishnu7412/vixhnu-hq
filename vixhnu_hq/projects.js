// ADD A NEW PROJECT: copy one block from { to }, paste it at the END of the list
// (after the last comma), then change the words and the code.
// Keep the word String.raw before the first backtick. Put your code between the two backticks.
// Newest project shows first on the site.

const projects = [
  {
    title: "Tiny Neuron (OR gate)",
    about: "One single neuron that teaches itself the OR gate using NumPy",
    tags: ["python", "numpy", "neural-net"],
    code: String.raw`
import numpy as np

# One single "neuron" that learns the OR gate:
# the answer is 1 if at least one of the two inputs is 1.


def squash(x):
    """Turns any number into a number between 0 and 1."""
    return 1 / (1 + np.exp(-x))


learning_rate = 0.1  # how big each learning step is

weight = np.random.randn(2)
bias = 0

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
right_answers = np.array([0, 1, 1, 1])

print("starting weights:", weight)

for step in range(1000):
    guess = squash(np.dot(inputs, weight) + bias)
    gap = right_answers - guess  # how wrong each guess is
    weight = weight + learning_rate * np.dot(inputs.T, gap)
    bias = bias + learning_rate * gap.sum()

    if step % 200 == 0:
        print(f"step {step}: guesses = {np.round(guess, 2)}")

final = squash(np.dot(inputs, weight) + bias)
print("\nfinal guesses:", np.round(final, 2))
print("rounded:      ", np.round(final).astype(int))
print("right answers:", right_answers)
input()
`
  },
  {
    title: "XOR Neural Network",
    about: "A small neural network with a hidden layer that learns XOR",
    tags: ["python", "numpy", "neural-net"],
    code: String.raw`
import numpy as np

# A tiny neural network that learns XOR:
# the answer is 1 only when the two inputs are different.
# Layout: 2 inputs -> 4 hidden neurons -> 1 output


def squash(x):
    """Turns any number into a number between 0 and 1."""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))  # clip keeps huge numbers safe


learning_rate = 0.5
hidden_size = 4

# layer 1: inputs -> hidden neurons
weight1 = np.random.randn(2, hidden_size)
bias1 = np.zeros(hidden_size)

# layer 2: hidden neurons -> output
weight2 = np.random.randn(hidden_size)
bias2 = 0

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
right_answers = np.array([0, 1, 1, 0])

for step in range(10000):
    # forward: make a guess
    hidden = squash(np.dot(inputs, weight1) + bias1)
    output = squash(np.dot(hidden, weight2) + bias2)

    # how wrong is the output?
    gap = right_answers - output
    output_push = gap * output * (1 - output)

    # how much did each hidden neuron cause that mistake?
    hidden_push = np.outer(output_push, weight2) * hidden * (1 - hidden)

    # learn: nudge every weight and bias
    weight2 = weight2 + learning_rate * np.dot(hidden.T, output_push)
    bias2 = bias2 + learning_rate * output_push.sum()
    weight1 = weight1 + learning_rate * np.dot(inputs.T, hidden_push)
    bias1 = bias1 + learning_rate * hidden_push.sum(axis=0)

    if step % 2000 == 0:
        print(f"step {step}: average mistake = {np.mean(gap ** 2):.4f}")

hidden = squash(np.dot(inputs, weight1) + bias1)
output = squash(np.dot(hidden, weight2) + bias2)
print("\nfinal answers:", np.round(output, 2))
print("rounded:      ", np.round(output).astype(int))
print("right answers:", right_answers)
input()
`
  },
  {
    title: "GPT69 Terminal Assistant",
    about: "A chat-style assistant: Wikipedia search, save text to a file, YouTube search",
    tags: ["python", "requests"],
    code: String.raw`
# GPT69: a small terminal assistant.
# It needs one extra package:  pip install requests
import webbrowser
from urllib.parse import quote, quote_plus

import requests

HEADERS = {"User-Agent": "GPT69/1.0 (personal learning project)"}


def default():
    """Open the YouTube home page."""
    webbrowser.open("https://youtube.com")


def search_yt(text):
    """Search YouTube for the text."""
    webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(text)}")


def clean_query(text):
    """Remove the command words (search / write) and keep the topic."""
    words = [w for w in text.split() if w.lower() not in ("search", "write")]
    return " ".join(words)


def search(text):
    """Get a short Wikipedia summary. Returns None if nothing is found."""
    query = clean_query(text)
    if not query:
        return None
    topic = quote(query.replace(" ", "_"))
    try:
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
            headers=HEADERS,
            timeout=10,
        )
    except requests.exceptions.RequestException:
        print("GPT69:- Slow internet connection, try again later!")
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json().get("extract")
    except ValueError:
        return None


def write_to_file(filename, text):
    done = False
    while not done:
        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"GPT69:- {text} \n writing operation in {filename} completed sir.")
            done = True
        except FileNotFoundError:
            print("GPT69:- Oops ! that isn't a valid file.")
            filename = input("GPT69:- try entering the filename again - ")
        except PermissionError:
            print("GPT69:- Oops ! that file isn't allowing me to write in it.")
            filename = input("GPT69:- try entering the another file's name - ")
        except IsADirectoryError:
            print("GPT69:- Oops ! that isn't a file.")
            filename = input("GPT69:- try entering a file name - ")


print("GPT69:- Hello sir, where should we begin today?")
print("GPT69:- (commands: search ..., write ..., yt, def, exit)")

while True:
    user_inp = input("you:- ").strip()
    words = user_inp.lower().split()

    if not words:
        continue

    if words[0] in ("exit", "quit"):
        print("GPT69:- Goodbye sir!")
        break

    elif "write" in words:
        user_content = search(user_inp)
        if user_content:
            filename = input("GPT69:- kindly enter the file's name sir- ")
            write_to_file(filename, user_content)
        else:
            print("GPT69:- Nothing to write sir!")

    elif "search" in words:
        user_content = search(user_inp)
        if user_content:
            print(f"GPT69:- {user_content}")
        else:
            print(f"GPT69:- I couldn't find anything about '{clean_query(user_inp)}' sir.")

    elif "yt" in words:
        data = input("You:- ")
        search_yt(data)

    elif "def" in words:
        default()

    else:
        print(f"GPT69:- I am really sorry sir but there's no such information \n available about {user_inp}.")

    print("GPT69:- what should we do next sir?")
`
  },
];
