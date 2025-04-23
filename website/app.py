import streamlit as st
import streamlit.components.v1 as components
import requests
import os
from music21 import stream, metadata, harmony, environment, note
import tempfile
import base64
import subprocess
from PIL import Image, ImageChops
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
import time

from chords_processing import ChordfromRoot

environment.set("musescoreDirectPNGPath", "/usr/bin/mscore")

st.set_page_config(
    page_title="Chord progression generator",
    page_icon=":musical_score:",
    initial_sidebar_state="expanded",
)

# Hide the Streamlit top toolbar (including the three-dot menu)
st.html(body="""
    <style>
        .stAppToolbar {display: none;}
    </style>
    """)

st.title(body = "Chord Progression Generator")

# User input for chord progression
st.subheader(body="Select your chords")

# Create a list to store the selected chords
if 'selected_chords' not in st.session_state:
    st.session_state.selected_chords = []

# List all possible chords
chord_roots = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#',]
#  'Ab', 'Bb', 'Db', 'Eb', 'Gb']
chord_qualities = ['', 'm', '7']

# Form to select a root note and chord quality
with st.form("chord_form"):
    # Step 1: User selects the root note
    root_note = st.selectbox(label="Select root note:", options=chord_roots)

    # Step 2: User selects the chord quality
    chord_quality = st.selectbox(label="Select chord quality:", options=["Major", "Minor", "Dominant 7th"])

    # Map quality names to corresponding symbols
    quality_map = {
        "Major": "",
        "Minor": "m",
        "Dominant 7th": "7"
    }

    # Create the chord
    selected_chord = root_note + quality_map[chord_quality]

    # Step 3: Add button to submit the chord
    submit_button = st.form_submit_button(label="Add chord")

    # Add the chord to the list if button is pressed
    if submit_button:
        st.session_state.selected_chords.append(selected_chord)
        st.success(body=f"Added chord: {selected_chord}")

# Display the list of selected chords
if st.session_state.selected_chords:
    st.write(f"Selected Chords: **{', '.join(st.session_state.selected_chords)}**")

input_chords = st.session_state.selected_chords

# User input for number of chords to generate
n_chords = st.number_input(
    label = "Number of chords to generate (1 to 30)",
    value = 15,
    min_value = 1,
    max_value = 30,
)

# User input for randomness
randomness = st.slider(
    label = "Chords complexity",
    min_value = 1,
    max_value = 10,
    value = 6
)

st.markdown(body = "## Prediction")

@st.cache_data
def call_api(
    input_chords: list,
    n_chords: int,
    randomness: int
    ):

    url = 'https://sammyeltakriti.com/predict'

    # Join the list of chords into a single comma-separated string
    song_param = ','.join(input_chords)

    parameters = {
        'song': song_param,
        'n_chords': n_chords,
        'randomness': randomness,
    }

    try:
        response = requests.get(url=url, params=parameters)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle connection errors and other issues
        print(f"Error: Unable to connect to the API: {e}")
        return None


def color_chords(chords, n_chords):
    """
    Colors chords dynamically based on n_chords value.
    """
    colored_chords = []
    for i, chord in enumerate(chords):
        if i < len(chords) - n_chords:
            colored_chords.append(f"<span style='color:black;'>{chord}</span>")
        else:
            colored_chords.append(f"<span style='color:green;'>{chord}</span>")
    return " ".join(colored_chords)


# Generate and display chords
if st.button(label="Generate Chords", type="primary"):
    # Check if the user has selected at least one chord
    if not st.session_state.selected_chords:
        st.warning("Please select at least one chord.")
    else:
        with st.spinner(text="Loading... Please wait while we generate the chords."):
            try:
                # Attempt to make the API call
                chords_predicted = call_api(input_chords, n_chords, randomness)
                if chords_predicted:
                    chords_predicted = chords_predicted.get('predicted chords', None)
            except KeyError as e:
                # Handle missing key in the response
                print(f"Error: The expected key was not found in the API response: {e}")
                chords_predicted = None
            except Exception as e:
                # Catch all other exceptions
                print(f"An unexpected error occurred: {e}")
                chords_predicted = None

        # Handle what happens if the API call fails
        if chords_predicted is None:
            st.error(body="Failed to retrieve predicted chords. Please try again later.")

        else:
            try:
                # Color chords
                styled_chords = color_chords(chords_predicted, n_chords)
                # Display result
                st.markdown(f'Predicted chords based on your input : <div>{styled_chords}</div>', unsafe_allow_html=True)

                with st.spinner(text="Generating sheet music..."):
                    song_stream = stream.Stream()

                    # Add chords and their names
                    for chord_input in chords_predicted:
                        chord_created = ChordfromRoot(chord_input).chordCreated
                        # Add a ChordSymbol for the chord name
                        chord_symbol = harmony.ChordSymbol(chord_input)
                        song_stream.append(chord_symbol)
                        song_stream.append(chord_created)

                    # # Save the musical score as an image
                    # fp = song_stream.write('lily.png', fp='chord_progression.png')

                    # Generate a unique identifier for the session
                    unique_id = uuid.uuid4().hex  # Generates a unique ID, e.g., '5b6e0c33e5f84c5f80a2a5d7e4c4c333'

                    # Define unique file paths within the static directory
                    static_dir = "static"
                    os.makedirs(static_dir, exist_ok=True) # Ensure static directory exists
                    path_name = f"song_stream_{unique_id}"
                    musicxml_path = os.path.join(static_dir, f"{path_name}.musicxml")
                    png_path = os.path.join(static_dir, f"{path_name}.png")
                    midi_path = os.path.join(static_dir, f"{path_name}.midi")

                    # Save the stream as MusicXML
                    song_stream.write(fmt='musicxml', fp=musicxml_path)

                    def remove_metadata_from_musicxml(file_path):
                        """
                        Removes metadata such as title and author from a MusicXML file.
                        """
                        # Parse the XML file
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        # Remove the <movement-title> element if it exists
                        for title in root.findall("movement-title"):
                            root.remove(title)

                        # Remove the <creator> element inside <identification>
                        for identification in root.findall("identification"):
                            for creator in identification.findall("creator"):
                                identification.remove(creator)

                        # Save the modified MusicXML file
                        tree.write(file_path, encoding="utf-8", xml_declaration=True)

                    # Remove metadata from the MusicXML file
                    remove_metadata_from_musicxml(musicxml_path)

                    # Generate the PNG file using MuseScore
                    subprocess.run(['xvfb-run', 'mscore', musicxml_path, '-o', png_path])

                    # Rename the generated file if it exists
                    generated_file = png_path.replace('.png', '-1.png')  # Adjust to MuseScore's naming
                    if os.path.exists(generated_file):
                        os.rename(generated_file, png_path)

                    # Crop the PNG file to remove blank space
                    if os.path.exists(png_path):
                        with Image.open(png_path) as img:
                            # Remove alpha channel for better cropping (convert to RGB)
                            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                                img = img.convert("RGB")

                            # Find the bounding box of non-white areas
                            bg_color = (255, 255, 255)  # Assume white background
                            diff = ImageChops.difference(img, Image.new(img.mode, img.size, bg_color))
                            bbox = diff.getbbox()

                            # Crop and save the image if a bounding box was found
                            if bbox:
                                cropped_img = img.crop(bbox)
                                cropped_img.save(png_path)
                            else:
                                print("No non-white areas detected. Image might already be blank.")

                    # Resize the image
                    with Image.open(png_path) as img:
                        resize_factor = 0.50
                        width, height = img.size
                        resized_image = img.resize((int(width * resize_factor), int(height * resize_factor)))
                        resized_image.save(png_path)

                    # Display the image in Streamlit
                    st.image(image=png_path, use_container_width=False)

                with st.spinner(text="Generating MIDI player..."):
                    # Save MIDI file
                    song_stream.write(fmt='midi', fp=midi_path)

                    # Embed the MIDI player in the Streamlit app
                    # See https://github.com/cifkao/html-midi-player
                    html_code = f"""
                    <script
                        src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0">
                    </script>
                    <midi-player
                        src="/{midi_path}"
                        sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/salamander"
                        visualizer="#myVisualizer">
                    </midi-player>
                    <midi-visualizer
                        type="piano-roll"
                        id="myVisualizer">
                    </midi-visualizer>
                    """

                    components.html(html=html_code, height=500)


            except Exception as e:
                st.error(f"An error occurred while processing the chords: {e}")
