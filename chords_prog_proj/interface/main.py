import numpy as np
import pandas as pd
import ast
import os
import json

from colorama import Fore, Style

from tensorflow.keras.utils import to_categorical

from chords_prog_proj.ml_logic.data import get_most_recent_file
from chords_prog_proj.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from chords_prog_proj.ml_logic.params import CHUNK_SIZE, DATASET_SIZE, VALIDATION_DATASET_SIZE, LOCAL_DATA_PATH, DATA_FILE
from chords_prog_proj.ml_logic.registry import get_model_version
from chords_prog_proj.ml_logic.preprocessor import preprocess

from chords_prog_proj.ml_logic.registry import load_model, save_model


def train(
    length = 12,
    number_of_samples = 5000
):
    """
    Train a new model on the full (already preprocessed) dataset.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set.
    """
    print("\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Get the processed CSV file with the most recent timestamp based on the naming pattern
    data_folder = os.path.join(os.getcwd(), LOCAL_DATA_PATH, "processed")
    file_pattern = "processed_data_*.csv"
    most_recent_file = get_most_recent_file(data_folder, file_pattern)

    if most_recent_file:
        print(f"Most recent processed CSV file: {most_recent_file}")
    else:
        print("No processed CSV files found.")

    df = pd.read_csv(most_recent_file)
    print(Fore.GREEN + f"\nLoaded DataFrame with shape {df.shape}" + Style.RESET_ALL)

    # Convert strings to lists
    df['chords'] = df['chords'].apply(ast.literal_eval)

    X = df['chords']

    # Create a dictionary which stores a unique token for each chord:
    # the key is the chord while the value is the corresponding token.
    chord_to_id = {}
    iter_ = 1
    for song in X:
        for chord in song:
            if chord in chord_to_id:
                continue
            chord_to_id[chord] = iter_
            iter_ += 1

    # Define the list of all notes and prefixes
    all_notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#', 'Ab', 'Bb', 'Db', 'Eb', 'Gb']
    prefix = ['m', '7', 'm7', 'm7b5', 'dim', 'dim7', 'maj', 'maj7', 'sus2', 'sus4', 'aug', 'add9', '6', 'm6', '9', 'm9', '11', 'm11', '13', 'm13', 'dim9', 'hdim7', '5']

    # Select chords that match the pattern of note + prefix or just the note itself
    selected_chords = {
        chord: value
        for chord, value in chord_to_id.items()
        if any(
            chord.startswith(note) and chord[len(note):] in prefix for note in all_notes
        ) or chord in all_notes  # Add this condition to select chords that are just the name of a note
    }

    chord_to_id = selected_chords

    # Reassign consecutive IDs to chords, keeping 'UNKNOWN' at 0
    chord_to_id_fixed = {'UNKNOWN': 0}
    for idx, chord in enumerate(chord_to_id.keys(), start=1):
        if chord != 'UNKNOWN':
            chord_to_id_fixed[chord] = idx

    # Replace the old chord_to_id with the fixed one
    chord_to_id = chord_to_id_fixed

    # Calculate vocab_size based on the fixed dictionary
    vocab_size = len(chord_to_id)

    # Save dictionary to json file
    with open('chord_to_id.json', 'w') as f:
        json.dump(chord_to_id, f)

    def get_X_y(song, length):
        """
        Function that, given a song (list of chords), returns:
            - a list of strings (list of chords) that corresponds to part of the song - this string should be of length N
            - the chord that follows the previous list of chords
        """
        if len(song) <= length:
            return None

        first_chord_idx = np.random.randint(0, len(song)-length)

        X_chords = song[first_chord_idx:first_chord_idx+length]
        y_chords = song[first_chord_idx+length]

        return X_chords, y_chords

    def create_dataset(songs, number_of_samples):
        """
        Function, that, based on the previous function and the loaded songs, generate a dataset X and y:
        - each sample of X is a list of chords
        - the corresponding y is the chord that comes just after in the input list of chords
        """

        X, y = [], []
        indices = np.random.randint(0, len(songs), size=number_of_samples)

        for idx in indices:
            ret = get_X_y(songs[idx], length=length)
            if ret is None:
                continue
            xi, yi = ret
            X.append(xi)
            y.append(yi)

        return X, y

    # Create X and y
    X, y = create_dataset(X, number_of_samples)

    # Split X and y in train and test data.
    len_ = int(0.7*len(X))
    chords_train = X[:len_]
    chords_test = X[len_:]
    y_train = y[:len_]
    y_test = y[len_:]

    # Based on the dictionary of chords to id, tokenize X and y (chords to numbers)
    X_train = [
        [chord_to_id.get(chord, chord_to_id['UNKNOWN']) for chord in song]
        for song in chords_train
    ]
    X_test = [
        [chord_to_id.get(chord, chord_to_id['UNKNOWN']) for chord in song]
        for song in chords_test
    ]

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train_token = [
        chord_to_id.get(chord, chord_to_id['UNKNOWN']) for chord in y_train
    ]
    y_test_token = [
        chord_to_id.get(chord, chord_to_id['UNKNOWN']) for chord in y_test
    ]

    # Ensure the tokens are within bounds of vocab_size
    max_token_train = max(y_train_token)
    max_token_test = max(y_test_token)

    if max_token_train >= vocab_size or max_token_test >= vocab_size:
        print(f"Warning: Some tokens exceed vocab_size ({vocab_size}). Max train token: {max_token_train}, Max test token: {max_token_test}")

    # Convert the tokenized outputs to one-hot encoded categories
    y_train_cat = to_categorical(y_train_token, num_classes=vocab_size)
    y_test_cat = to_categorical(y_test_token, num_classes=vocab_size)

    # Load model
    model = None
    model = load_model()  # production model

    # Model params
    learning_rate = 0.001
    epochs = 15
    batch_size = 256
    patience = 15
    output_dim = 50

    # Initialize model
    if model is None:
        model = initialize_model(vocab_size=vocab_size, output_dim=output_dim)

    print(Fore.BLUE + "\nCompile and fit preprocessed validation data..." + Style.RESET_ALL)

    # Compile and train the model
    model = compile_model(model, learning_rate)
    model, history = train_model(
        model,
        X_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    # Return the highest value of the validation accuracy
    metrics_val = np.max(history.history['val_accuracy'])
    print(f"\n✅ Trained with accuracy: {round(metrics_val, 2)}")

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Dataset size
        number_of_samples=number_of_samples,

        # Package behavior
        context="train",

        # Data source
        model_version=get_model_version(),
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(accuracy=metrics_val))

    # Evaluate on the test set
    print(Fore.BLUE + "\nEvaluate on the test set..." + Style.RESET_ALL)
    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test_cat)
    #accuracy = metrics_dict["accuracy"]

    return metrics_val


def pred(song: list = None,
         n_chords=4,
         randomness=1,
         model=None) -> list:
    """
    Make a prediction using the latest trained model
    """
    print("\n⭐️ Use case: predict")

    # Load dictionary chords_to_id
    with open("chord_to_id.json", "r") as json_file:
        chord_to_id = json.load(json_file)

    # Create dictionary id_to_chord
    id_to_chord = {v: k for k, v in chord_to_id.items()}
    # Remove "UNKNOWN" value
    id_to_chord.pop(0)

    if model is None:
        model = load_model()

    if song is None:
        song = ['G', 'D', 'G', 'D', 'Am', 'F', 'Em', 'F#',]

    def get_predicted_chord(song, randomness=1):
        # Convert chords to numbers
        song_convert = [chord_to_id[chord] for chord in song]

        # Return an array of size vocab_size, with the probabilities
        pred = model.predict([song_convert], verbose=0)

        # Return the index of nth probability
        pred_class = np.argsort(np.max(pred, axis=0))[-randomness]

        if pred_class == 0:
            pred_class = 1

        # Turn the index into a chord
        try:
            pred_chord = id_to_chord[pred_class]
        except:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)

        return pred_chord

    def repeat_prediction(song, n_chords, randomness=1):
        song_tmp = song
        if randomness == 1:
            for i in range(n_chords):
                predicted_chord = get_predicted_chord(song_tmp, randomness=randomness)
                song_tmp.append(predicted_chord)
        else:
            random_list = np.random.randint(low=1, high=randomness+1, size=n_chords, dtype=int)
            for i in range(n_chords):
                predicted_chord = get_predicted_chord(song_tmp, randomness=random_list[i])
                song_tmp.append(predicted_chord)

        return song_tmp

    chord_pred = get_predicted_chord(song=song, randomness=randomness)
    print(f"\n✅ predicted next chord witn randomness {randomness}: ", chord_pred)

    n_chords_pred = repeat_prediction(song=song, n_chords=n_chords, randomness=randomness)
    print(f"\n✅ predicted {n_chords} next chords witn randomness {randomness}: ", n_chords_pred)

    return n_chords_pred


if __name__ == '__main__':
    # train()
    new_song = ["G", "D"]
    pred(
        song = new_song,
        n_chords= 10,
        randomness = 1
        )
    # evaluate()
