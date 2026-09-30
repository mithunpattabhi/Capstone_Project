# Adding Audio to Our Anomaly Detection Model

Hey! Since the STGCN model is working really well for the visual side of things, it makes total sense to tackle the low-light/darkness problem next. In places like hospital wards at night, the video feed is going to be pretty useless. But even if we can't see an anomaly (like a fall or someone shouting), we can definitely *hear* it.

Here's the plan on how we can pull audio from our existing video feeds, analyze the decibel levels (and other sound features), and hook it all up to our current model.

## 1. What Are We Actually Listening For?

When something goes wrong in the dark, it usually makes a noise. We can break this down into two main approaches:

*   **Sudden Volume Spikes (The Quick Method):** 
    The easiest thing to check is if something is just abnormally loud. We can calculate the Root Mean Square (RMS) energy or straight-up decibel (dB) levels. If the ward is usually quiet at night and there's a sudden massive spike, that's an instant red flag.
*   **Sound Signatures (The Smarter Method):** 
    Just looking at volume might give us false alarms (like a door slamming vs. someone falling). To fix this, we can extract Mel-Frequency Cepstral Coefficients (MFCCs). Think of MFCCs as a unique "fingerprint" for sounds. It helps the system tell the difference between footsteps, a human voice in distress, or a heavy thud on the floor.

## 2. The Tech Stack: How Do We Do It?

We don't need any crazy new hardware since the audio is already baked into the video files. Here's what we'll use:

*   **Extracting the Audio:** 
    We'll use `FFmpeg`. It's a rock-solid tool that we can call right from a Python script to quickly strip the `.wav` audio tracks out of our video datasets.
*   **Processing the Sound:** 
    We'll use `librosa`. It's basically the go-to library for audio in Python. It can easily load our extracted audio, calculate the dB levels, and pull those MFCC sound fingerprints I mentioned.

## 3. Stitching It Together (Multimodal Fusion)

This is the big question: how do we make our existing STGCN model and this new audio system play nice together? 

The best approach here is **Late Fusion**. Instead of trying to force audio data directly into the STGCN architecture (which would be a massive headache and might break what's already working), we let them work side-by-side.

1.  **Independent Scoring:** The STGCN watches the video and spits out a confidence score (e.g., "I'm 80% sure this is an anomaly"). The audio script listens to the sound and gives its own score.
2.  **Smart Weighting:** We build a simple logic gate on top of them. If we detect that the video feed is mostly black/too dark, we tell the system to practically ignore the STGCN's score and rely heavily on the audio score. 

This way, our video model stays exactly as it is, but we get a massive safety net when the lights go out.

## 4. Game Plan / Next Steps

Here is how we can start building this out practically:

1.  **Extract the Audio:** Write a quick script to loop through our video data and dump out the audio using `FFmpeg`.
2.  **The Decibel Test:** Write a script using `librosa` to plot the volume levels of normal events vs. anomalous events. We might find that a simple dB threshold is all we really need.
3.  **Train a Classifier (Optional but recommended):** If the simple volume test isn't accurate enough, we can train a really lightweight model (like a Random Forest) on the MFCC features to actually classify *what* the sound is.
4.  **Write the Fusion Logic:** Combine the outputs of both models into a final script that makes the ultimate "Anomaly / No Anomaly" decision based on lighting conditions.

Let me know what you think of this approach! It's the cleanest way to make our model robust without having to start from scratch.
