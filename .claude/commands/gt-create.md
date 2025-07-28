# gt create and submit

## Context

### 
[​](https://graphite.dev/docs/command-reference#gt-create-%5Bname%5D)
`gt create [name]`
Create a new branch stacked on top of the current branch and commit staged changes. If no branch name is specified, generate a branch name from the commit message. If your working directory contains no changes, an empty branch will be created. If you have any unstaged changes, you will be asked whether you’d like to stage them.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-4)
arguments
`[name] (optional)`The name of the new branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-7)
flags
`--ai`Automatically AI-generate the branch name and the commit message (if unset) `-a, --all`Stage all unstaged changes before creating the branch, including to untracked files. `-i, --insert`Insert this branch between the current branch and its child. If there are multiple children, prompts you to select which should be moved onto the new branch. `-m, --message`Specify a commit message. `--no-ai`Do not automatically AI-generate the branch name and the commit message. Takes precedence over —ai. `-p, --patch`Pick hunks to stage before committing. `-u, --update`Stage all updates to tracked files before creating the branch. `-v, --verbose`Show unified diff between the HEAD commit and what would be committed at the bottom of the commit message template. If specified twice, show in addition the unified diff between what would be committed and the worktree files, i.e. the unstaged changes to tracked files.

### 
[​](https://graphite.dev/docs/command-reference#gt-submit)
`gt submit`
Idempotently force push all branches from trunk to the current branch to GitHub, creating or updating distinct pull requests for each. Validates that branches are properly restacked before submitting, and fails if there are conflicts. Blocks force pushes to branches that overwrite branches that have changed since you last submitted or got them. Opens an interactive prompt that allows you to input pull request metadata. `gt ss` is a default alias for `gt submit --stack`.
#### 
[​](https://graphite.dev/docs/command-reference#flags-25)
flags
`--ai`Automatically AI-generate title and description for all PRs. Only works when creating new PRs. If —edit, use the generated metadata as starting points. `--always`Always push updates, even if the branch has not changed. Can be helpful for fixing an inconsistent Graphite stack view on Web/GitHub resulting from downtime/a bug. `--branch`Which branch to run this command from. Defaults to the current branch. `--cli`Edit PR metadata via the CLI instead of on web. `--comment`Add a comment on the PR with the given message. `-c, --confirm`Reports the PRs that would be submitted and asks for confirmation before pushing branches and opening/updating PRs. If either of —no-interactive or —dry-run is passed, this flag is ignored. `-d, --draft`If set, all new PRs will be created in draft mode. `--dry-run`Reports the PRs that would be submitted and terminates. No branches are restacked or pushed and no PRs are opened or updated. `-e, --edit`Input metadata for all PRs interactively. If neither —edit nor —no-edit is passed, only prompts for new PRs. `--edit-description`Input the PR description interactively. Default only prompts for new PRs. Takes precedence over —no-edit. `--edit-title`Input the PR title interactively. Default only prompts for new PRs. Takes precedence over —no-edit. `-f, --force`Force push: overwrites the remote branch with your local branch. Otherwise defaults to —force-with-lease. `--ignore-out-of-sync-trunk`Perform the submit operation even if the trunk branch is out of sync with its upstream branch. This can lead to incorrect metadata being used during the submit. `-m, --merge-when-ready`If set, marks all PRs being submitted as merge when ready, which will let them automatically merge as soon as all merge requirements are met. `--no-ai`Don’t use AI to generate any PR fields. Takes precedence over —ai. `-n, --no-edit`Don’t edit any PR fields inline. Takes precedence over —edit. `--no-edit-description`Don’t prompt for the PR description. Takes precedence over —edit-description and —edit. `--no-edit-title`Don’t prompt for the PR title. Takes precedence over —edit-title and —edit. `-p, --publish`If set, publishes all PRs being submitted. `--rerequest-review`Rerequest review from current reviewers. `--restack`Restack branches before submitting. If there are conflicts, output the branch names that could not be restacked `-r, --reviewers`If set without an argument, prompt to manually set reviewers. Alternatively, accepts a comma separated string of reviewers `-s, --stack`Submit descendants of the current branch in addition to its ancestors. `--target-trunk`Which trunk to open PRs against on remote. Defaults to the target trunk for the current local trunk (defined in `gt config`), or the current local trunk if no target trunk is configured. `-t, --team-reviewers`Comma separated list of team slugs. You can either pass “slug” to this flag or “org/slug” to the reviewers flag. Will enable the —reviewers prompt if set without arguments. `-u, --update-only`Only push branches and update PRs for branches that already have PRs open. `-v, --view`Open the PR in your browser after submitting. `-w, --web`Open a web browser to edit PR metadata, even if no new PRs are being created or if configured to edit PR metadata via the CLI.

## 커밋 메시지 작성 가이드

- 형식: **<type>(<scope>): <subject>**
- type: `feat | fix | docs | style | refactor | test | chore`
- subject: 50자 이하, 현재형 동사 + 한글
- 본문(선택): 왜(WHY)·무엇(WHAT)·영향(IMPACT)을 네-줄 이하로
1. git staging에 있는 코드 변경 내역을 검사해서 위의 커밋 메시지 작성 가이드에 따라 커밋 메시지 작성
2. gt create -m "1번에서 작성한 커밋 메시지 내용" 으로 실행
3. gt submit --stack 실행