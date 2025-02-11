import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key="YOUR_API_KEY")

def analyze_conversation(messages):
    """
    Analyzes a conversation and returns sentiment classification with a quantified mental state.
    
    Parameters:
        messages (list): List of user messages (strings) in the conversation.
    
    Returns:
        dict: Processed emotions, mood scores, and overall mental state summary.
    """

    # Construct the prompt dynamically based on user messages
    conversation_text = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])
    
    prompt = f"""
    Analyze the following conversation and classify each message as Happy, Sad, Anxious, or Calm.
    Also, provide a mood score from -1 (very negative) to 1 (very positive) for each message.

    Conversation:
    {conversation_text}

    Return the response as a JSON object with:
    - 'analysis': A list of dictionaries for each message with fields ('text', 'mood', 'score')
    - 'overall_mood': The most frequent mood in the conversation.
    - 'average_mood_score': The mean score across all messages.
    - 'mood_distribution': A breakdown of each mood type count.

    Example Output:
    {{
        "analysis": [
            {{"text": "I'm feeling amazing today!", "mood": "Happy", "score": 0.9}},
            {{"text": "I'm so stressed about work.", "mood": "Anxious", "score": -0.7}}
        ],
        "overall_mood": "Anxious",
        "average_mood_score": -0.4,
        "mood_distribution": {{"Happy": 1, "Sad": 0, "Anxious": 1, "Calm": 0}}
    }}
    """

    # Send request to OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    return response.choices[0].message.content  # Return API response

# Example conversation messages
conversation = [
    "I'm feeling amazing today!",
    "I don't know if I can do this, I'm really stressed.",
    "I had a great time with my friends at lunch!",
    "Work has been exhausting, I just need a break.",
    "Just sitting here, enjoying some peace and quiet."
]

# Analyze the conversation
result = analyze_conversation(conversation)

# Print the response
print(result)
