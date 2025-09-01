from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import TypedDict, Dict,List, Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

model_name = "gemma3:latest"
model = ChatOllama(model=model_name,temperature=0)

class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]

def read_email(state: EmailState) -> EmailState:
    email = state["email"]

    print(f"Alfred is processing the email from '{email['sender']}' with subject '{email['subject']}'")

    return {}

def classify_email(state: EmailState) -> EmailState:
    email = state["email"]
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()
    print("Alfred's analysis:", response_text)

    is_spam = "spam" in response_text and "not spam" not in response_text

    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
        print("Spam reason:", spam_reason)
    
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "feedback", "other"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    new_messages = state.get("messages", []) + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response.content}]

    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState) -> EmailState:
    if state["is_spam"]:
        print("Alfred is marking the email as spam.")
    else:
        print("Alfred is not marking the email as spam.")
    return {}

def draft_response(state: EmailState) -> EmailState:
    if state["is_spam"]:
        return {"email_draft": None}

    email = state["email"]
    category = state["email_category"] or "general"

    prompt = f"""
    As Alfred the butler, draft a polite and professional response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    print("Alfred's draft response:", response.content)

    new_messages = state.get("messages", []) + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response.content}]

    return {
        "email_draft": response.content,
        "messages": new_messages
    }

def notify_mr_wayne(state: EmailState) -> EmailState:
    email = state["email"]
    print(f"Notifying Mr. Wayne about the email from '{email['sender']}' with subject '{email['subject']}'")
    if state["is_spam"]:
        print("This email was marked as spam and will not be shown to Mr. Wayne.")
    else:
        print("This email is legitimate.")
        if state["email_draft"]:
            print("Alfred has drafted a response for Mr. Wayne to review:")
            print(state["email_draft"])
        else:
            print("No draft response was created.")
    return {}

def route_email(state: EmailState) -> EmailState:
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"
    

email_graph = StateGraph(EmailState)
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_wayne", notify_mr_wayne)

email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")
email_graph.add_conditional_edges("classify_email", route_email, {
    "spam": "handle_spam",
    "legitimate": "draft_response"
})
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_wayne")
email_graph.add_edge("notify_mr_wayne", END)

compiled_graph = email_graph.compile()

# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Wayne, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# Process the legitimate email
print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# Process the spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

langfuse_handler = CallbackHandler()
legitimate_result = compiled_graph.invoke(
    input={"email": legitimate_email, "is_spam": None, "spam_reason": None, "email_category": None, "draft_response": None, "messages": []},
    config={"callbacks": [langfuse_handler]}
)