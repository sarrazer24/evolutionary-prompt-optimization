# Contribution guide

## Branch naming

- Member 1: `feature/data-evaluation`
- Member 2: `feature/llm-prompt-engineering`
- Member 3: `feature/genetic-algorithm`
- Member 4: `feature/experiments-analysis`

## Workflow

1. Pull the latest `main`.
2. Create or update your own branch.
3. Work only in your owned folders unless a shared change is agreed.
4. Run tests and check imports before committing.
5. Commit with a clear message.
6. Push your branch and open a Pull Request.
7. Ask another member to review before merging into `main`.

## Commit examples

```bash
git add src/llm src/prompts
git commit -m "Add local LLM interface and prompt operators"
git push -u origin feature/llm-prompt-engineering
```

## Rules

- Never commit Kaggle tokens, Hugging Face tokens, passwords or `.env` files.
- Never commit model weights or large dataset files.
- Do not change shared interfaces without telling the team.
- Do not merge untested code directly into `main`.
- Keep notebooks focused; move reusable code into `src/`.
