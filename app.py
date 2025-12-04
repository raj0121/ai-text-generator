import streamlit as st
import random

# Configure the page
st.set_page_config(
    page_title="AI Text Generator with Sentiment",
    page_icon="✨",
    layout="wide"
)

# Title and description
st.title("✨ AI Text Generator with Sentiment Analysis")
st.markdown("Generate paragraphs matching your prompt's sentiment (Positive, Negative, or Neutral)")

# Sample texts for demo mode
DEMO_TEXTS = {
    "😊 Positive": [
        "The future of technology holds incredible promise for humanity. With each breakthrough, we move closer to solving some of our most pressing challenges. The potential for AI to enhance healthcare, education, and environmental sustainability is truly inspiring. As we continue to develop these technologies responsibly, we can look forward to a world where human potential is amplified.",
        "Advancements in artificial intelligence bring hope and excitement for what lies ahead. The collaborative spirit driving innovation today ensures that we are building tools that empower rather than replace, that connect rather than divide. This journey of discovery reminds us of our shared humanity and the boundless possibilities when we work together.",
        "The progress we are witnessing in renewable energy technology is truly remarkable. Solar and wind power are becoming more efficient and affordable each year, making clean energy accessible to more people worldwide. This transition not only helps combat climate change but also creates new economic opportunities and improves public health."
    ],
    "😠 Negative": [
        "The rapid development of artificial intelligence raises serious concerns about ethical boundaries and societal impact. As algorithms grow more powerful, questions about privacy erosion, job displacement, and unchecked automation demand urgent attention. The concentration of AI capabilities in few hands threatens to exacerbate existing inequalities.",
        "The environmental crisis continues to worsen despite global awareness. Deforestation, plastic pollution, and carbon emissions are reaching critical levels, with devastating consequences for ecosystems and human health. The slow pace of policy implementation and corporate resistance to change are major obstacles to meaningful progress.",
        "Social media platforms have created unprecedented challenges for mental health and social cohesion. The spread of misinformation, online harassment, and addictive design patterns are harming individuals and societies. Despite growing evidence of these negative effects, regulation remains inadequate and implementation inconsistent."
    ],
    "😐 Neutral": [
        "Artificial intelligence represents a significant technological development with both potential benefits and challenges. Current applications range from healthcare diagnostics to autonomous systems, each with distinct implications for society. The evolution of AI continues to prompt discussions about ethical frameworks and regulatory approaches.",
        "Climate change mitigation requires a balanced approach considering scientific evidence, economic factors, and social equity. Various strategies exist, including renewable energy adoption, carbon capture technologies, and behavioral changes. Each approach has different implementation timelines, costs, and effectiveness profiles.",
        "Remote work has transformed traditional employment structures, offering both flexibility and new challenges. Studies show mixed results regarding productivity, work-life balance, and team collaboration in remote settings. Organizations continue to experiment with hybrid models to optimize outcomes across different industries."
    ]
}

# Function to check dependencies
def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        from transformers import pipeline
        return True, None
    except ImportError as e:
        missing = []
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        try:
            from transformers import pipeline
        except ImportError:
            missing.append("transformers")
        
        return False, missing

# Check dependencies
deps_available, missing_packages = check_dependencies()

# Show warning if dependencies missing
if not deps_available:
    st.warning(f"⚠️ Some dependencies are missing: {', '.join(missing_packages)}")
    
    with st.expander("🔧 How to fix this"):
        st.markdown("""
        ### Installation Instructions
        
        **For Local Installation:**
        ```bash
        pip install torch transformers streamlit
        ```
        
        **For Streamlit Cloud, update `requirements.txt`:**
        ```txt
        streamlit==1.28.0
        transformers==4.35.0
        torch==2.1.0
        --extra-index-url https://download.pytorch.org/whl/cpu
        ```
        
        **Alternative (simpler):**
        ```txt
        streamlit==1.28.0
        transformers[torch]==4.35.0
        ```
        """)
    
    st.info("Running in **Demo Mode** with sample texts. Install dependencies for full AI functionality.")

# Load AI models if dependencies are available
if deps_available:
    try:
        from transformers import pipeline
        
        @st.cache_resource(show_spinner=False)
        def load_ai_models():
            """Load and cache AI models"""
            with st.spinner("Loading AI models (first time may take a minute)..."):
                # Load sentiment analysis model
                sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                
                # Load text generation model
                text_model = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_length=300
                )
                
                return sentiment_model, text_model
        
        # Load models
        sentiment_analyzer, text_generator = load_ai_models()
        
    except Exception as e:
        st.error(f"Failed to load AI models: {str(e)}")
        st.info("Running in Demo Mode instead.")
        deps_available = False
        sentiment_analyzer = None
        text_generator = None
else:
    sentiment_analyzer = None
    text_generator = None

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Sentiment options
    if deps_available and sentiment_analyzer:
        sentiment_option = st.radio(
            "Sentiment Detection:",
            ["Auto Detect", "Manual Select"],
            help="Auto Detect uses AI to analyze sentiment from your prompt"
        )
    else:
        sentiment_option = "Manual Select"
        st.caption("Auto Detect requires AI models")
    
    # Manual sentiment selection
    if sentiment_option == "Manual Select":
        selected_sentiment = st.selectbox(
            "Choose Sentiment:",
            ["😊 Positive", "😠 Negative", "😐 Neutral"],
            index=0
        )
    
    # Output length
    output_length = st.slider(
        "Output Length (words):",
        min_value=50,
        max_value=300,
        value=150,
        step=10,
        help="Approximate number of words in generated text"
    )
    
    # Creativity control
    creativity = st.slider(
        "Creativity Level:",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values = more creative/random, Lower values = more focused"
    )
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
    1. Enter a prompt
    2. Choose or detect sentiment
    3. Adjust length & creativity
    4. Click Generate!
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Prompt")
    user_prompt = st.text_area(
        "What would you like to write about?",
        placeholder="Example: 'The impact of artificial intelligence on education' or 'Challenges in climate change mitigation'...",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("💡 Example Prompts")
    examples = [
        "Select an example...",
        "The joy of learning new skills",
        "Frustrations with urban traffic",
        "Benefits of renewable energy",
        "Challenges in online education",
        "Future of space exploration"
    ]
    
    selected_example = st.selectbox("Try these:", examples)
    
    if selected_example != "Select an example...":
        user_prompt = selected_example

# Generate button
generate_clicked = st.button("🚀 Generate Text", type="primary", use_container_width=True)

# Process generation when button is clicked
if generate_clicked:
    if not user_prompt or user_prompt.strip() == "":
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Analyzing and generating text..."):
            
            # STEP 1: Determine sentiment
            if sentiment_option == "Auto Detect" and sentiment_analyzer:
                try:
                    # Analyze sentiment using AI
                    sentiment_result = sentiment_analyzer(user_prompt[:512])[0]
                    detected_label = sentiment_result['label']
                    confidence_score = sentiment_result['score']
                    
                    # Map to our sentiment categories
                    label_mapping = {
                        "POSITIVE": "😊 Positive",
                        "NEGATIVE": "😠 Negative",
                        "LABEL_0": "😐 Neutral",
                        "LABEL_1": "😊 Positive"
                    }
                    
                    determined_sentiment = label_mapping.get(detected_label, "😐 Neutral")
                    
                    # Display sentiment results
                    sentiment_col1, sentiment_col2 = st.columns(2)
                    with sentiment_col1:
                        st.metric("Detected Sentiment", determined_sentiment)
                    with sentiment_col2:
                        st.metric("Confidence", f"{confidence_score:.1%}")
                    
                    final_sentiment = determined_sentiment
                    
                except Exception as e:
                    st.warning(f"Could not detect sentiment automatically: {str(e)}")
                    final_sentiment = selected_sentiment if 'selected_sentiment' in locals() else "😐 Neutral"
            else:
                final_sentiment = selected_sentiment if 'selected_sentiment' in locals() else "😐 Neutral"
            
            # STEP 2: Generate text
            if deps_available and text_generator:
                try:
                    # Prepare sentiment-guided prompt
                    sentiment_guides = {
                        "😊 Positive": "Write an optimistic and positive text about: ",
                        "😠 Negative": "Write a critical and analytical text about: ",
                        "😐 Neutral": "Write a balanced and objective text about: ",
                        "Positive": "Write an optimistic and positive text about: ",
                        "Negative": "Write a critical and analytical text about: ",
                        "Neutral": "Write a balanced and objective text about: "
                    }
                    
                    guide_prefix = sentiment_guides.get(final_sentiment, "Write about: ")
                    enhanced_prompt = guide_prefix + user_prompt
                    
                    # Calculate appropriate max length
                    max_tokens = min(output_length * 2, 500)
                    
                    # Generate text with AI
                    generated_output = text_generator(
                        enhanced_prompt,
                        max_length=max_tokens,
                        temperature=creativity,
                        do_sample=True,
                        top_p=0.9,
                        num_return_sequences=1
                    )
                    
                    ai_generated_text = generated_output[0]['generated_text']
                    
                    # Remove the guiding prefix if present
                    if enhanced_prompt in ai_generated_text:
                        ai_generated_text = ai_generated_text.replace(enhanced_prompt, "").strip()
                    
                    # Display the generated text
                    st.markdown("---")
                    st.subheader("📄 AI Generated Text")
                    
                    st.text_area(
                        "Your generated text:",
                        ai_generated_text,
                        height=300,
                        label_visibility="collapsed"
                    )
                    
                    # Show statistics
                    actual_word_count = len(ai_generated_text.split())
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Word Count", actual_word_count)
                    with stats_col2:
                        st.metric("Sentiment", final_sentiment)
                    with stats_col3:
                        st.metric("Creativity", f"{creativity:.1f}")
                    
                    # Copy option
                    if st.button("📋 Copy to Clipboard"):
                        st.code(ai_generated_text)
                        st.success("Text ready to copy! Select and press Ctrl+C")
                    
                except Exception as e:
                    st.error(f"AI generation failed: {str(e)}")
                    st.info("Falling back to demo mode for this generation.")
                    deps_available = False
            
            # DEMO MODE: Show sample text if AI is not available
            if not deps_available or not text_generator:
                st.info("🎭 **Demo Mode**: Showing sample text based on your inputs")
                
                # Select appropriate demo text
                demo_options = DEMO_TEXTS.get(final_sentiment, DEMO_TEXTS["😐 Neutral"])
                base_demo_text = random.choice(demo_options)
                
                # Adjust length
                words = base_demo_text.split()
                if len(words) > output_length:
                    words = words[:output_length]
                    adjusted_text = " ".join(words) + "..."
                else:
                    adjusted_text = base_demo_text
                
                # Add creativity variation
                if creativity > 0.8:
                    variations = [
                        " This represents ongoing developments in the field.",
                        " Further research will explore these dimensions.",
                        " The implications continue to be studied by experts."
                    ]
                    if random.random() > 0.3:
                        adjusted_text += random.choice(variations)
                
                # Display demo text
                st.markdown("---")
                st.subheader("📄 Sample Generated Text (Demo)")
                
                st.text_area(
                    "Sample output (install dependencies for AI generation):",
                    adjusted_text,
                    height=300,
                    label_visibility="collapsed"
                )
                
                # Demo stats
                demo_word_count = len(adjusted_text.split())
                
                demo_col1, demo_col2 = st.columns(2)
                with demo_col1:
                    st.metric("Words", demo_word_count)
                with demo_col2:
                    st.metric("Mode", "Demo")
                
                st.caption("🔧 Install dependencies for AI-powered text generation")

# Footer with information
st.markdown("---")
with st.expander("ℹ️ About this Application"):
    st.markdown("""
    ### AI Text Generator with Sentiment Analysis
    
    **How it works:**
    1. **Sentiment Analysis**: Uses DistilBERT to detect if your prompt is Positive, Negative, or Neutral
    2. **Text Generation**: Uses GPT-2 to generate coherent paragraphs
    3. **Sentiment Alignment**: Guides the AI to match the detected/selected sentiment
    
    **Features:**
    - ✅ Real-time sentiment detection
    - ✅ AI-powered text generation
    - ✅ Manual sentiment override
    - ✅ Adjustable output length (50-300 words)
    - ✅ Creativity control (0.1-1.0)
    - ✅ Demo mode for testing
    - ✅ Error handling and fallbacks
    
    **Models Used:**
    - 🤖 Sentiment Analysis: `distilbert-base-uncased-finetuned-sst-2-english`
    - ✍️ Text Generation: `gpt2` (or demo mode)
    
    **Note**: First run downloads AI models (~500MB). Subsequent runs are faster!
    """)

with st.expander("🛠️ Technical Requirements"):
    st.markdown("""
    ### Dependencies
    
    **Required for AI functionality:**
    ```txt
    streamlit==1.28.0
    transformers==4.35.0
    torch==2.1.0
    ```
    
    **For Streamlit Cloud deployment, use:**
    ```txt
    streamlit==1.28.0
    transformers==4.35.0
    torch==2.1.0
    --extra-index-url https://download.pytorch.org/whl/cpu
    ```
    
    **Installation commands:**
    ```bash
    # Local installation
    pip install streamlit transformers torch
    
    # Or using requirements.txt
    pip install -r requirements.txt
    ```
    
    **To run locally:**
    ```bash
    streamlit run app.py
    ```
    """)

st.caption("Built with ❤️ using Streamlit & HuggingFace Transformers | Assessment Submission")
