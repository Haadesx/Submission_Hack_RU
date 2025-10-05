## Inspiration
Our inspiration for Hack Tracks stemmed from a shared observation: in a world that's more connected than ever, genuine emotional support often feels out of reach. We saw friends and community members struggle with the high costs, stigma, and long waiting times associated with traditional mental health care. We believe that technology should serve humanity, and we were inspired by the potential to bridge this accessibility gap. We asked ourselves: what if we could create a first line of support that is immediate, personalized, and universally accessible? By combining the analytical power of AI with the universal, therapeutic language of music, we were inspired to build a tool that could meet people exactly where they are, emotionally, and offer a moment of relief and understanding.

## What it does
Hack Tracks is an AI-powered emotional wellness companion. At its core, the platform performs three key functions in real-time:

1.  **🎭 Real-Time Emotion Detection:** Using a device's camera, our platform analyzes the user's facial expressions through a `ResNet-34` Convolutional Neural Network. The model identifies one of seven core emotions (happy, sad, angry, neutral, surprised, fearful, or disgusted) with a 68% accuracy rate.

2.  **🎵 AI-Curated Music Therapy:** Based on the detected emotion, Hack Tracks instantly connects to the Spotify API to generate a personalized, location-aware music playlist. This can either match the user's current mood to provide validation or introduce a different vibe to help shift their emotional state.

3.  **🤖 Conversational AI Support:** Simultaneously, the user is greeted by a supportive AI therapist. This conversational agent provides empathetic dialogue and support, delivered with natural voice narration to create a more human and comforting experience.

In essence, Hack Tracks detects how you feel and immediately provides actionable, supportive content to enhance your emotional well-being, 24/7.

## How we built it
Hack Tracks was built by integrating several key technologies into a cohesive platform:

* **Emotion Detection Engine:** The heart of our project is a `ResNet-34` Convolutional Neural Network (CNN) built using PyTorch. We trained this model on a public dataset of facial expressions, fine-tuning it to classify seven different emotional states from a live video feed.

* **Music Curation:** We utilized the **Spotify API** for all music-related features. After our model outputs an emotion, our backend sends a request to the Spotify API with carefully crafted parameters (e.g., genre, mood, tempo, and location-based trends) to dynamically create a unique playlist for the user.

* **AI Therapist:** The conversational support system is powered by a large language model. We integrated it with a Text-to-Speech (TTS) engine to provide the natural voice narration, making the interaction feel more personal and less robotic.

* **Frontend & Integration:** We developed a simple user interface that connects the camera input to our AI model, displays the detected emotion, and presents the generated Spotify playlist and the chat interface for the AI therapist. All components were designed to work together seamlessly to create a fluid user experience.

## Challenges we ran into
Building a multi-faceted AI project in a short timeframe came with several challenges:

1.  **Model Accuracy and Real-World Conditions:** Achieving reliable accuracy with our emotion detection model was a significant hurdle. Real-world variables like poor lighting, varied head poses, and subtle expressions initially led to inconsistent results. It took considerable effort in data augmentation and model tuning to reach our current accuracy of 68%.

2.  **API Integration Latency:** Synchronizing the three core pillars—emotion detection, music generation, and AI chat—in real-time was difficult. We initially faced latency issues where the playlist or chat response would lag behind the emotion detection, breaking the feeling of an instantaneous response. We worked on optimizing our code with asynchronous calls to make the experience smoother.

3.  **Computational Resources:** Processing a live video stream for facial detection and emotion classification is computationally intensive. Our initial model was too slow to run effectively in real-time. We had to optimize the model's architecture and the preprocessing pipeline to ensure it could run efficiently without draining system resources.

## Accomplishments that we're proud of
Despite the challenges, we are incredibly proud of what we achieved:

* **A Fully Integrated Prototype:** Our biggest accomplishment is creating a functional, end-to-end prototype that successfully integrates three complex technologies: computer vision, a third-party music API, and conversational AI.

* **Achieving 68% Accuracy:** For a hackathon project, developing and training a CNN model to achieve a 68% accuracy rate in real-world emotion detection is something we are very proud of. This proves our core concept is viable.

* **Human-Centric Design:** We successfully built a tool that isn't just technologically impressive, but is designed with a clear, human-centric purpose: to make mental health support more accessible and less intimidating. The positive feedback on the seamless user experience has been very rewarding.

## What we learned
This project was a tremendous learning experience for our entire team.

* **Technical Skills:** We gained deep, practical experience in the entire machine learning pipeline—from data collection and preprocessing to training, evaluating, and deploying a computer vision model. We also sharpened our skills in working with complex APIs and integrating disparate services into a single application.

* **The Nuances of Affective Computing:** We learned that human emotion is incredibly complex and nuanced. Building a model to understand it taught us about the limitations of current technology and the ethical considerations required when developing tools that interact with a user's emotional state.

* **Rapid Prototyping and Teamwork:** The hackathon environment forced us to be agile, prioritize features effectively, and collaborate seamlessly under pressure. We learned how to break down a complex vision into manageable tasks and execute them efficiently.

## What's next for HackTracks
We see Hack Tracks as more than just a hackathon project; we believe it has the potential to become a powerful tool for daily emotional wellness. Our next steps include:

* **Improving Model Accuracy and Scope:** Our top priority is to increase the model's accuracy to over $85\%$. We plan to achieve this by training it on a larger, more diverse dataset to better recognize emotions across different demographics and in various environmental conditions. We also aim to detect more nuanced emotions beyond the basic seven.

* **Enhanced Personalization:** We want to add a feedback loop where users can confirm or correct the detected emotion, allowing the model to learn and personalize its responses over time. We also plan to track mood patterns (with user consent) to provide insights and long-term wellness analytics.

* **Expanding Therapeutic Features:** We intend to integrate other wellness modules triggered by the user's emotional state, such as guided breathing exercises for anxiety, short meditations for stress, or positive affirmation prompts for sadness.

* **Mobile Application Development:** Our ultimate goal is to develop a fully-fledged, cross-platform mobile application, making Hack Tracks an accessible and pocket-sized emotional wellness companion for everyone.