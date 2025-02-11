import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key="")

def analyze_conversation(conversation):
    """
    Analyzes a full conversation, extracting only user messages for emotion tracking.
    
    Parameters:
        conversation (list of dicts): List of messages with roles ('user' or 'assistant').

    Returns:
        dict: Processed emotions, mood scores, and overall mental state summary.
    """

    # Extract only user messages
    user_messages = [msg["content"] for msg in conversation if msg["role"] == "user"]
    
    # Construct the input dynamically for GPT
    conversation_text = "\n".join([f"User: {msg}" for msg in user_messages])

    prompt = f"""
    Analyze the following conversation. Only consider the 'User' messages.
    Classify each message as Happy, Sad, Anxious, or Calm. Also, assign a mood score (-1 to 1).

    Conversation:
    {conversation_text}

    Return the response as a JSON object with:
    - 'analysis': A list of dictionaries for each user message with ('text', 'mood', 'score')
    - 'overall_mood': The most frequent mood.
    - 'average_mood_score': The mean score across all user messages.
    - 'mood_distribution': A breakdown of mood frequencies.

    Example Output:
    {{
        "analysis": [
            {{"text": "I'm feeling amazing today!", "mood": "Happy", "score": 0.9}},
            {{"text": "I'm really anxious about my exam.", "mood": "Anxious", "score": -0.7}}
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

# Example AI-User conversation
conversation = [
    {"role": "user", "content": "I feel exhausted after today’s work."},
    {"role": "assistant", "content": "That sounds tough. Did anything specific make it stressful?"},
    {"role": "user", "content": "Yeah, my manager keeps giving me extra tasks at the last minute."},
    {"role": "assistant", "content": "That’s frustrating. Have you tried talking to them about it?"},
    {"role": "user", "content": "I did, but they don’t seem to care. It’s just overwhelming."},
    {"role": "assistant", "content": "I understand. Maybe taking a break would help a little?"},
    {"role": "user", "content": "I wish I could. There's just too much to do."}
]

# Analyze the conversation
result = analyze_conversation(conversation)

# Print the response
print(result)
