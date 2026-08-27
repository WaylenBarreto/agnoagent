from __future__ import annotations

import asyncio
import json
import os
from email.utils import parseaddr
from pathlib import Path

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.google.gmail import GmailTools

# qwen3 uses Ollama's structured <tool_call> format, which Agno can execute.
# llama3.2 returned tool-call JSON as plain text in this demo.
MODEL_ID = "qwen3:4b"
APP_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = Path(os.getenv("GMAIL_CREDENTIALS_PATH", str(APP_DIR / "credentials.json")))
TOKEN_PATH = Path(os.getenv("GMAIL_TOKEN_PATH", str(APP_DIR / "gmail_token.json")))


def gmail_model() -> Ollama:
    """Use tool calls without Qwen's extended reasoning phase."""
    return Ollama(
        id=MODEL_ID,
        request_params={"think": False},
        options={"num_predict": 400},
        timeout=90,
    )


def create_gmail_toolkit(*, write: bool) -> GmailTools:
    """Create a raw Gmail toolkit for the agent to execute actual Gmail methods."""
    if not CREDENTIALS_PATH.is_file():
        raise RuntimeError(
            "Gmail OAuth credentials file was not found. Download the Desktop OAuth "
            "client JSON from Google Cloud, save it as credentials.json, or set "
            "GMAIL_CREDENTIALS_PATH to its path."
        )

    scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    if write:
        scopes = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.compose",
        ]

    return GmailTools(
        credentials_path=str(CREDENTIALS_PATH),
        token_path=str(TOKEN_PATH),
        scopes=scopes,
        get_latest_emails=True,
        get_unread_emails=True,
        get_emails_from_user=True,
        get_starred_emails=True,
        get_emails_by_context=True,
        get_emails_by_date=True,
        get_emails_by_thread=True,
        search_emails=True,
        mark_email_as_read=False,
        mark_email_as_unread=False,
        star_email=False,
        unstar_email=False,
        archive_email=False,
        apply_label=False,
        remove_label=False,
        delete_custom_label=False,
        get_message=True,
        get_thread=True,
        list_drafts=write,
        get_draft=write,
        create_draft_email=write,
        send_email=False,
        send_email_reply=False,
        update_draft=write,
    )


def authorize_gmail() -> None:
    """Run Google OAuth once and save gmail_token.json for later demo runs."""
    if not CREDENTIALS_PATH.is_file():
        raise RuntimeError(
            "Gmail OAuth credentials file was not found. Save the downloaded Desktop "
            "OAuth JSON as credentials.json before running this script."
        )

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    toolkit = GmailTools(
        credentials_path=str(CREDENTIALS_PATH),
        token_path=str(TOKEN_PATH),
        scopes=[
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        get_latest_emails=True,
        create_draft_email=True,
        send_email=False,
        send_email_reply=False,
    )
    result = json.loads(toolkit.get_latest_emails(count=1))
    if "error" in result:
        raise RuntimeError(f"Gmail authorization failed: {result['error']}")
    if not TOKEN_PATH.is_file():
        raise RuntimeError(
            f"Gmail authorization did not create token file at {TOKEN_PATH}."
        )
    print(f"Gmail OAuth connected. Token saved to {TOKEN_PATH}.")

async def demo_read_only():
    print("\n" + "=" * 60)
    print("DEMO 1: Read-Only Gmail Access")
    print("=" * 60)

    gmail = create_gmail_toolkit(write=False)
    result = json.loads(gmail.search_emails(query="is:unread newer_than:3d", count=10))
    emails = result.get("emails", [])
    if not emails:
        print("No unread emails were found in the last 3 days.")
        return

    email_context = "\n\n".join(
        f"From: {email['from']}\nSubject: {email['subject']}\nBody: {email['body'][:1500]}"
        for email in emails
    )
    agent = Agent(model=gmail_model(), markdown=True)
    print(f"Read {len(emails)} unread Gmail message(s). Summarizing with {MODEL_ID}...\n")
    await agent.aprint_response(
        "Group these unread emails by sender and summarize what each person is asking for. "
        "Do not invent details.\n\n" + email_context,
        stream=True,
    )


async def demo_read_write():
    print("\n" + "=" * 60)
    print("DEMO 2: Read-Write Gmail Access")
    print("=" * 60)

    gmail = create_gmail_toolkit(write=True)
    result = json.loads(gmail.search_emails(query="is:unread newer_than:3d", count=1))
    emails = result.get("emails", [])
    if not emails:
        print("No unread email is available to create a follow-up draft.")
        return

    email = emails[0]
    recipient = parseaddr(email["from"])[1]
    if not recipient:
        raise RuntimeError("Could not determine a reply recipient from the latest unread email.")

    agent = Agent(model=gmail_model(), markdown=True)
    draft_prompt = (
        "Write a brief, professional follow-up reply. Return only the email body; "
        "do not include a subject or greeting placeholders.\n\n"
        f"Original sender: {email['from']}\nSubject: {email['subject']}\n"
        f"Email body: {email['body'][:2000]}"
    )
    draft_body = str((await agent.arun(draft_prompt)).content).strip()
    subject = email["subject"]
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    draft_result = json.loads(gmail.create_draft_email(
        to=recipient,
        subject=subject,
        body=draft_body,
        thread_id=email.get("thread_id"),
        message_id=email.get("id"),
    ))
    if "error" in draft_result:
        raise RuntimeError(f"Gmail draft creation failed: {draft_result['error']}")
    print("Draft created successfully. No email was sent.")


async def main():
    authorize_gmail()
    await demo_read_only()
    await demo_read_write()


if __name__ == "__main__":
    asyncio.run(main())
