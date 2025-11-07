# TECHDEVELOPERS-PROJECTS
📈 Stock Sentiment Analysis
🔍 Overview

This project analyzes the sentiment of stock-related news headlines to gauge market sentiment and predict stock price movements.

⚙️ Installation & Setup
# Clone the repository
git clone https://github.com/priyanka981026-hub/TECHDEVELOPERS-PROJECTS.git
cd TECHDEVELOPERS-PROJECTS/my_stock_sentiment

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run stock1.py

🧪 Usage

Input stock-related news headlines.

The app processes and displays sentiment analysis results.

🛠️ Technologies

Python

Streamlit

pandas

scikit-learn


🌐 My Web Browser
🔍 Overview

This project is a simple web browser built using PyQt5.
It includes basic functionalities like back, forward, reload, home, stop, and a URL bar for navigation.

⚙️ Installation & Setup
# Clone the repository
git clone https://github.com/priyanka981026-hub/TECHDEVELOPERS-PROJECTS.git
cd TECHDEVELOPERS-PROJECTS/streamlit/text_summarizer_project

# Install dependencies
pip install PyQt5 PyQtWebEngine

# Run the application
python web_browser.py

🧪 Usage

Open the app and enter a URL in the address bar.

Use the navigation toolbar buttons to go back, forward, reload, stop, or go home.

The window title updates dynamically based on the current webpage.

🧪 Features

🔙 Back and 🔜 Forward navigation

🔄 Reload page

🏠 Home button (default: Google)

⛔ Stop page loading

🌐 URL bar for direct navigation

📝 Dynamic window title reflecting the current page

🎨 Technologies Used
Technology	Purpose
Python 3.x	Core programming language
PyQt5	GUI development
PyQtWebEngine	Embedded web browser

👩‍💻 Author

Priyanka Yadav
Python Developer

📫 Connect with me:
LinkedIn | GitHub


🧠 Priyanka - Human Face Generator (AI + Deep Learning Project)

An AI-powered Streamlit web app that generates realistic human faces using StyleGAN2-ADA trained on the CelebA Dataset (and your own images).

🌟 Project Overview

This project demonstrates how Generative Adversarial Networks (GANs) can be used to create highly realistic human faces from random noise vectors.
It allows users to upload their own dataset, preprocess images, load pretrained models, and even fine-tune the model on custom data.

🎯 Features

✅ Upload and preprocess your own celebrity or personal face dataset
✅ Load a pretrained StyleGAN2-ADA model (FFHQ)
✅ Generate AI-based human faces with custom seeds
✅ Supports GPU acceleration (CUDA)
✅ View generated faces directly in the Streamlit dashboard
✅ Option to fine-tune the model for personalized results

🧰 Tech Stack
Technology	Purpose
Python	Core programming
Streamlit	Interactive web app
PyTorch	Deep learning framework
StyleGAN2-ADA	GAN model for image generation
Torchvision & PIL	Image processing
NumPy & Matplotlib	Data handling and visualization
Librosa (optional)	Audio/image preprocessing
tqdm, shutil, requests	Utility operations
⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/priyanka981026-hub/Human-Face-Generator.git
cd Human-Face-Generator


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run "C:\Users\Acer Aspire 3\streamlit\streamlit\text_summarizer_project\streamlit_facegen.py"


4️⃣ Upload Dataset and Generate Faces

Follow on-screen steps to upload images, preprocess, load the model, and generate faces.

🚀 GPU Support

The app automatically detects your GPU:

device = "cuda" if torch.cuda.is_available() else "cpu"


If available, face generation runs 5–10× faster using GPU acceleration.

🖼️ Output Example

Upload your dataset

Generate faces

Visualize them directly on the Streamlit dashboard

Example generated output 👇


📚 References

StyleGAN2-ADA Official Repo (NVIDIA)

CelebA Dataset

Streamlit Documentation

🤝 Contributing

Pull requests and suggestions are welcome!
If you find this project helpful, don’t forget to ⭐ the repo.

👩‍💻 Author

Priyanka Yadav
📩 Python Developer | Data Analyst | Deep Learning Enthusiast
🔗 GitHub
 | LinkedIn

🏷️ License

This project is released under the MIT License — free to use and modify.
