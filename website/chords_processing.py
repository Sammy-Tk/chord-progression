from music21 import chord, note, pitch

class ChordfromRoot:
    def __init__(self, chord_input: str):
        """
        Creates a chord based on the input string.

        Args:
        chord_input (str): The chord string (e.g., "B", "B-m", "C#maj7").
        """
        # Replace flat symbols "Bb" -> "B-""
        self.chord_input = chord_input.replace("b", "-")

        self.root_note, self.suffix = self.extract_root_and_suffix(self.chord_input)

        self.chordCreated = self.create_chord(self.root_note, self.suffix)

    def extract_root_and_suffix(self, chord_input: str):
        """
        Logic for extracting root and suffix
        """

        # Valid root notes (the 12 note names ['A', 'B-', 'B', 'C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'G#'])
        names_of_notes = [note.Note('A').transpose(n).name for n in range(12)]

        try:
            # Check for 2-character root notes (e.g. B-, C#)
            if chord_input[:2] in names_of_notes:
                root_note = chord_input[:2]
                suffix = chord_input[2:]  # Rest of the string after the root note
            # Check for 1-character root notes (e.g. B, C)
            elif chord_input[:1] in names_of_notes:
                root_note = chord_input[:1]
                suffix = chord_input[1:]  # Rest of the string after the root note
            else:
                raise ValueError(f"Invalid chord root in input: {chord_input}./nAccepted values:{names_of_notes}")
        except ValueError as e:
            print(e)
            root_note, suffix = None, None  # Default values for invalid input

        # Create music21 note object
        root_note = note.Note(root_note)

        return root_note, suffix

    def create_chord(self, root_note, suffix) -> chord.Chord:
        """
        Creates a chord based on the input strings using a dictionary of chord intervals.
        Supports major, minor, diminished, augmented, and seventh chords.

        Returns:
        music21.chord.Chord: A music21 chord object representing the requested chord.
        """
        # Dictionary mapping chord types to their intervals
        CHORD_INTERVALS = {
            "maj": [0, 4, 7],          # Major chord
            "m": [0, 3, 7],            # Minor chord
            "dim": [0, 3, 6],          # Diminished chord
            "aug": [0, 4, 8],          # Augmented chord
            "maj7": [0, 4, 7, 11],     # Major seventh chord
            "m7": [0, 3, 7, 10],       # Minor seventh chord
            "7": [0, 4, 7, 10],        # Dominant seventh chord
            "dim7": [0, 3, 6, 9],      # Diminished seventh chord
            "m7b5": [0, 3, 6, 10],     # Half-diminished seventh chord (minor 7 flat 5)
        }

        # Default to major chord if no valid suffix is found
        intervals = CHORD_INTERVALS.get(suffix, CHORD_INTERVALS["maj"])

        # Generate the chord pitches based on the root and intervals
        chord_pitches = [root_note.transpose(interval) for interval in intervals]

        # Create the music21 chord object
        new_chord = chord.Chord(chord_pitches)
        new_chord.duration.quarterLength = 1  # Set default duration as a quarter note

        return new_chord

    def transpose(self, interval: int):
        # Transpose logic
        pass
