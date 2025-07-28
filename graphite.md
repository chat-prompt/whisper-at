[Graphite home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-black.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-white.svg)](https://graphite.dev)
Search...
⌘K
##### Get started
  * [Overview](https://graphite.dev/docs/get-started)


##### Setup & configuration
  * Authenticate with GitHub
  * Configure GitHub repository settings
  * Install & authenticate the CLI


##### Diamond (AI code review)
  * [Overview](https://graphite.dev/docs/diamond)
  * [Setup & configuration](https://graphite.dev/docs/diamond-setup)
  * [Review comments](https://graphite.dev/docs/diamond-review-comments)
  * [Customization](https://graphite.dev/docs/diamond-customization)


##### PR workflows
  * [Pull Request Inbox](https://graphite.dev/docs/use-pr-inbox)
  * Review pull requests
  * Merge pull requests


##### Stacking (Graphite CLI)
  * [Overview](https://graphite.dev/docs/cli-overview)
  * [Quick Start](https://graphite.dev/docs/cli-quick-start)
  * [Command Reference](https://graphite.dev/docs/command-reference)
  * [Command Cheatsheet](https://graphite.dev/docs/cheatsheet)
  * [Configure The CLI](https://graphite.dev/docs/configure-cli)
  * [GT MCP](https://graphite.dev/docs/gt-mcp)
  * Basic tutorials
  * Advanced tutorials


##### Integrations
  * [Overview](https://graphite.dev/docs/integrations)
  * VS Code Extension
  * [Menu Bar App (Mac)](https://graphite.dev/docs/menu-bar-app)
  * [Slack Notifications](https://graphite.dev/docs/slack-notifications)
  * [Linear](https://graphite.dev/docs/linear)


##### Repository management
  * Merge Queue
  * [Automations](https://graphite.dev/docs/automations)
  * Insights


##### Administration
  * Privacy & security
  * [User permissions](https://graphite.dev/docs/graphite-admin)
  * Billing & plans
  * [GitHub Enterprise Server](https://graphite.dev/docs/github-enterprise-server)


##### References
  * FAQs
  * [CLI Changelog](https://graphite.dev/docs/cli-changelog)
  * Graphite CLI v1 command names
  * Learn to stack
  * Evaluating Graphite
  * [LLM-Friendly Documentation](https://graphite.dev/docs/ai-ingestion)


  * [App](https://app.graphite.dev/)


[Graphite home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-black.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-white.svg)](https://graphite.dev)
Search...
⌘KAsk AI
  * [App](https://app.graphite.dev/)


Search...
Navigation
Stacking (Graphite CLI)
Command Reference
Stacking (Graphite CLI)
# Command Reference
This reference documents every command and flag available in Graphite’s command-line interface.
Follow the [installation guide](https://graphite.dev/docs/install-the-cli) to set up the Graphite CLI.
## 
[​](https://graphite.dev/docs/command-reference#global-flags)
Global flags
`--help`Show help for a command. `--allCommands`This is not printed with the global help, but if passed to gt —help —all, will print out the full list of command help. `--cwd`Working directory in which to perform operations. `--debug`Write debug output to the terminal. `--interactive`Enable interactive features like prompts, pagers, and editors. Enabled by default. Disable with `--no-interactive`. `--verify`Enable git hooks. Enabled by default. Disable with `--no-verify`. `--quiet`Minimize output to the terminal. Implies `--no-interactive`.
## 
[​](https://graphite.dev/docs/command-reference#available-commands)
Available commands
### 
[​](https://graphite.dev/docs/command-reference#gt-abort)
`gt abort`
Aborts the current Graphite command halted by a rebase conflict.
#### 
[​](https://graphite.dev/docs/command-reference#flags)
flags
`-f, --force`Do not prompt for confirmation; abort immediately.
### 
[​](https://graphite.dev/docs/command-reference#gt-absorb)
`gt absorb`
Amend staged changes to the relevant commits in the current stack. Relevance is calculated by checking the changes in each commit downstack from the current commit, and finding the first commit that each staged hunk (consecutive lines of changes) can be applied to deterministically. If there is no clear commit to absorb a hunk into, it will not be absorbed. Prompts for confirmation before amending the commits, and restacks the branches upstack of the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-2)
flags
`-a, --all`Stage all unstaged changes before absorbing. Unlike create and modify, this will not include untracked files, as file creations would never be absorbed. `-d, --dry-run`Print which commits the hunks would be absorbed into, but do not actually absorb them. `-f, --force`Do not prompt for confirmation; apply the hunks to the commits immediately. `-p, --patch`Pick hunks to stage before absorbing.
### 
[​](https://graphite.dev/docs/command-reference#gt-add-%5Bargs-%5D)
`gt add [args..]`
git add passthrough
#### 
[​](https://graphite.dev/docs/command-reference#arguments)
arguments
`[args] (optional)`git add arguments
### 
[​](https://graphite.dev/docs/command-reference#gt-aliases)
`gt aliases`
Edit your command aliases.
#### 
[​](https://graphite.dev/docs/command-reference#flags-3)
flags
`--legacy`Append legacy aliases to your configuration. See <https://graphite.dev/docs/legacy-alias-preset> for more details. `--reset`Reset your alias configuration.
### 
[​](https://graphite.dev/docs/command-reference#gt-auth)
`gt auth`
Add your auth token to enable Graphite CLI to create and update your PRs on GitHub.
#### 
[​](https://graphite.dev/docs/command-reference#flags-4)
flags
`-t, --token`Auth token. Get it from: <https://app.graphite.dev/activate>
### 
[​](https://graphite.dev/docs/command-reference#gt-bottom)
`gt bottom`
Switch to the branch closest to trunk in the current stack.
### 
[​](https://graphite.dev/docs/command-reference#gt-changelog)
`gt changelog`
Show the Graphite CLI changelog.
### 
[​](https://graphite.dev/docs/command-reference#gt-checkout-%5Bbranch%5D)
`gt checkout [branch]`
Switch to a branch. If no branch is provided, opens an interactive selector.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-2)
arguments
`[branch] (optional)`The branch to checkout.
#### 
[​](https://graphite.dev/docs/command-reference#flags-5)
flags
`-a, --all`Show branches across all configured trunks in interactive selection. `-u, --show-untracked`Include untracked branches in interactive selection. `-s, --stack`Only show ancestors and descendants of the current branch in interactive selection. `-t, --trunk`Checkout the current trunk.
### 
[​](https://graphite.dev/docs/command-reference#gt-cherry-pick-%5Bargs-%5D)
`gt cherry-pick [args..]`
git cherry-pick passthrough
#### 
[​](https://graphite.dev/docs/command-reference#arguments-3)
arguments
`[args] (optional)`git cherry-pick arguments
### 
[​](https://graphite.dev/docs/command-reference#gt-children)
`gt children`
Show the children of the current branch.
### 
[​](https://graphite.dev/docs/command-reference#gt-completion)
`gt completion`
Set up `bash` or `zsh` tab completion.
### 
[​](https://graphite.dev/docs/command-reference#gt-config)
`gt config`
Configure the Graphite CLI.
### 
[​](https://graphite.dev/docs/command-reference#gt-continue)
`gt continue`
Continues the most recent Graphite command halted by a rebase conflict.
#### 
[​](https://graphite.dev/docs/command-reference#flags-6)
flags
`-a, --all`Stage all changes before continuing.
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
[​](https://graphite.dev/docs/command-reference#gt-dash)
`gt dash`
Opens your Graphite dashboard.
### 
[​](https://graphite.dev/docs/command-reference#gt-delete-%5Bname%5D)
`gt delete [name]`
Delete a branch and its Graphite metadata (local-only). Children will be restacked onto the parent branch. If the branch is not merged or closed, prompts for confirmation.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-5)
arguments
`[name] (optional)`The name of the branch to delete. If no branch is provided, opens an interactive selector.
#### 
[​](https://graphite.dev/docs/command-reference#flags-8)
flags
`-f, --force`Delete the branch even if it is not merged or closed.
### 
[​](https://graphite.dev/docs/command-reference#gt-demo-%5Bdemoname%5D)
`gt demo [demoName]`
Run interactive demos in any repo to learn how to use the Graphite CLI. This will teach you how to create pull requests & stacks with Graphite. Usage:
  1. gt demo pull-request: Learn how to create a PR
  2. gt demo stack: Learn how to create a stack of PRs


#### 
[​](https://graphite.dev/docs/command-reference#arguments-6)
arguments
`[demoName] (optional)`Demo to run
### 
[​](https://graphite.dev/docs/command-reference#gt-docs)
`gt docs`
Show the Graphite CLI docs.
### 
[​](https://graphite.dev/docs/command-reference#gt-down-%5Bsteps%5D)
`gt down [steps]`
Switch to the parent of the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-9)
flags
`-n, --steps`The number of levels to traverse downstack.
### 
[​](https://graphite.dev/docs/command-reference#gt-feedback-%5Bmessage%5D)
`gt feedback [message]`
Post a string directly to the maintainers’ Slack so they can drown in praise, factor in your feedback, laugh at your jokes, cry at your insults, or fall victim to Slack injection attacks.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-7)
arguments
`[message] (optional)`Positive or constructive feedback for the Graphite team! Jokes are chill too.
#### 
[​](https://graphite.dev/docs/command-reference#flags-10)
flags
`-d, --with-debug-context`Include logs from the past 24 hours in your feedback. This can help us understand what’s going on in your repo.
### 
[​](https://graphite.dev/docs/command-reference#gt-fish)
`gt fish`
Set up `fish` tab completion.
### 
[​](https://graphite.dev/docs/command-reference#gt-fold)
`gt fold`
Fold a branch’s changes into its parent, update dependencies of descendants of the new combined branch, and restack. This is useful when you have a branch that is no longer needed and you want to combine its changes with its parent branch. This command does not perform any action on GitHub or the remote repository. If you fold a branch with an open pull request, you will need to manually close the pull request.
#### 
[​](https://graphite.dev/docs/command-reference#flags-11)
flags
`-k, --keep`Keeps the name of the current branch instead of using the name of its parent.
### 
[​](https://graphite.dev/docs/command-reference#gt-get-%5Bbranch%5D)
`gt get [branch]`
For a given branch or PR number, sync branches from trunk to the given branch from remote, prompting the user to resolve any conflicts. If the branch passed to get already exists locally, any local branches upstack of the branch are also synced; to opt out of this behavior, use the —downstack flag. Note that remote-only branches upstack of the branch are not currently synced. If no branch is provided, sync the current stack.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-8)
arguments
`[branch] (optional)`Branch or PR number to get from remote.
#### 
[​](https://graphite.dev/docs/command-reference#flags-12)
flags
`-d, --downstack`When syncing a branch that already exists locally, don’t sync upstack branches. `-f, --force`Overwrite all fetched branches with remote source of truth `--restack`Restack any branches in the stack that can be restacked without conflicts (true by default; skip with —no-restack).
### 
[​](https://graphite.dev/docs/command-reference#gt-guide-%5Btitle%5D)
`gt guide [title]`
Read extended guides on how to use the gt program.
### 
[​](https://graphite.dev/docs/command-reference#gt-info-%5Bbranch%5D)
`gt info [branch]`
Display information about the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-9)
arguments
`[branch] (optional)`The branch to show info for. Defaults to the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-13)
flags
`-b, --body`Show the PR body, if it exists. `-d, --diff`Show the diff between this branch and its parent. Takes precedence over patch. `-p, --patch`Show the changes made by each commit. `-s, --stat`Show a diffstat instead of a full diff. Modifies either —patch or —diff. If neither is passed, implies —diff.
### 
[​](https://graphite.dev/docs/command-reference#gt-init)
`gt init`
Initialize Graphite in this repository by selecting a trunk branch. Can also be used to change the trunk branch of the repository.
#### 
[​](https://graphite.dev/docs/command-reference#flags-14)
flags
`--reset`Untrack all branches. `--trunk`The name of your trunk branch. If no name is passed, you will be prompted to select one interactively.
### 
[​](https://graphite.dev/docs/command-reference#gt-log-%5Bcommand%5D)
`gt log [command]`
Commands that log your stacks. Has three forms, `gt log`, `gt log short`, and `gt log long`.
  * `gt log long` ignores all options and displays a graph of the commit ancestry of all branches.
  * `gt log` and `gt log short` display all tracked branches and their dependency relationships.

The difference between the latter two is that `gt log` displays more information about each branch. `gt ls` and `gt ll` are default aliases for `gt log short` and `gt log long` respectively.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-10)
arguments
`[command] (optional)`The format to use. If not provided, `gt log` is assumed.
#### 
[​](https://graphite.dev/docs/command-reference#flags-15)
flags
`-a, --all`Show branches across all configured trunks. `--classic`Use the old short logging style, which runs out of screen real estate more quickly. Other options will not work in classic mode. `-r, --reverse`Print the log upside down. Handy when you have a lot of branches! `-u, --show-untracked`Include untracked branched in the log. `-s, --stack`Only show ancestors and descendants of the current branch. `-n, --steps`Only show this many levels upstack and downstack. Implies —stack.
### 
[​](https://graphite.dev/docs/command-reference#gt-merge)
`gt merge`
Merge the pull requests associated with all branches from trunk to the current branch via Graphite.
#### 
[​](https://graphite.dev/docs/command-reference#flags-16)
flags
`-c, --confirm`Asks for confirmation before merging branches. Prompts for confirmation if the local branches differ from remote, regardless of the value of this flag. `--dry-run`Reports the PRs that would be merged and terminates. No branches are merged.
### 
[​](https://graphite.dev/docs/command-reference#gt-modify)
`gt modify`
Modify the current branch by amending its commit or creating a new commit. Automatically restacks descendants. If you have any unstaged changes, you will be asked whether you’d like to stage them.
#### 
[​](https://graphite.dev/docs/command-reference#flags-17)
flags
`-a, --all`Stage all changes before committing. `-c, --commit`Create a new commit instead of amending the current commit. If this branch has no commits, this command always creates a new commit. `-e, --edit`If passed, open an editor to edit the commit message. When creating a new commit, this flag is ignored. `--interactive-rebase`Ignore all other flags and start a git interactive rebase on the commits in this branch. `-m, --message`The message for the new or amended commit. If passed, no editor is opened. `-p, --patch`Pick hunks to stage before committing. `-u, --update`Stage all updates to tracked files before committing. `-v, --verbose`Show unified diff between the HEAD commit and what would be committed at the bottom of the commit message template. If specified twice, show in addition the unified diff between what would be committed and the worktree files, i.e. the unstaged changes to tracked files.
### 
[​](https://graphite.dev/docs/command-reference#gt-move)
`gt move`
Rebase the current branch onto the target branch and restack all of its descendants. If no branch is passed in, opens an interactive selector.
#### 
[​](https://graphite.dev/docs/command-reference#flags-18)
flags
`-a, --all`Show branches across all configured trunks in interactive selection. `-o, --onto`Branch to move the current branch onto. `--source`Branch to move (defaults to current branch).
### 
[​](https://graphite.dev/docs/command-reference#gt-parent)
`gt parent`
Show the parent of the current branch.
### 
[​](https://graphite.dev/docs/command-reference#gt-pop)
`gt pop`
Delete the current branch but retain the state of files in the working tree.
### 
[​](https://graphite.dev/docs/command-reference#gt-pr-%5Bbranch%5D)
`gt pr [branch]`
Opens the pull request page for a branch or PR number. If no branch is passed, the current branch’s PR is opened.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-11)
arguments
`[branch] (optional)`A branch name or PR number to open.
#### 
[​](https://graphite.dev/docs/command-reference#flags-19)
flags
`--stack`Open the stack page.
### 
[​](https://graphite.dev/docs/command-reference#gt-rebase-%5Bargs-%5D)
`gt rebase [args..]`
git rebase passthrough
#### 
[​](https://graphite.dev/docs/command-reference#arguments-12)
arguments
`[args] (optional)`git rebase arguments
### 
[​](https://graphite.dev/docs/command-reference#gt-rename-%5Bname%5D)
`gt rename [name]`
Rename a branch and update metadata referencing it. If no branch name is supplied, you will be prompted for a new branch name. Note that this removes any association to a pull request, as GitHub pull request branch names are immutable.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-13)
arguments
`[name] (optional)`The new name for the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-20)
flags
`-f, --force`Allow renaming a branch that is already associated with an open GitHub pull request.
### 
[​](https://graphite.dev/docs/command-reference#gt-reorder)
`gt reorder`
Reorder branches between trunk and the current branch, restacking all of their descendants. Opens an editor where you can reorder branches by moving around a line corresponding to each branch.
### 
[​](https://graphite.dev/docs/command-reference#gt-reset-%5Bargs-%5D)
`gt reset [args..]`
git reset passthrough
#### 
[​](https://graphite.dev/docs/command-reference#arguments-14)
arguments
`[args] (optional)`git reset arguments
### 
[​](https://graphite.dev/docs/command-reference#gt-restack)
`gt restack`
Ensure each branch in the current stack has its parent in its Git commit history, rebasing if necessary. If conflicts are encountered, you will be prompted to resolve them via an interactive Git rebase.
#### 
[​](https://graphite.dev/docs/command-reference#flags-21)
flags
`--branch`Which branch to run this command from. Defaults to the current branch. `--downstack`Only restack this branch and its ancestors. `--only`Only restack this branch. `--upstack`Only restack this branch and its descendants.
### 
[​](https://graphite.dev/docs/command-reference#gt-restore-%5Bargs-%5D)
`gt restore [args..]`
git restore passthrough
#### 
[​](https://graphite.dev/docs/command-reference#arguments-15)
arguments
`[args] (optional)`git restore arguments
### 
[​](https://graphite.dev/docs/command-reference#gt-revert-%5Bsha%5D)
`gt revert [sha]`
Create a branch that reverts a commit on the trunk branch. Currently experimental.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-16)
arguments
`[sha]`The commit to revert.
#### 
[​](https://graphite.dev/docs/command-reference#flags-22)
flags
`-e, --edit`Edit the commit message.
### 
[​](https://graphite.dev/docs/command-reference#gt-split)
`gt split`
Split the current branch into multiple single-commit branches.
#### 
[​](https://graphite.dev/docs/command-reference#flags-23)
flags
`-c, --commit, --by-commit`Split by commit - slice up the history of this branch. `-h, --hunk, --by-hunk`Split by hunk - split into new single-commit branches.
### 
[​](https://graphite.dev/docs/command-reference#gt-squash)
`gt squash`
Squash all commits in the current branch into a single commit and restack upstack branches.
#### 
[​](https://graphite.dev/docs/command-reference#flags-24)
flags
`--edit`Modify the existing commit message. `-m, --message`The updated message for the commit. `-n, --no-edit`Don’t modify the existing commit message. Takes precedence over —edit
### 
[​](https://graphite.dev/docs/command-reference#gt-submit)
`gt submit`
Idempotently force push all branches from trunk to the current branch to GitHub, creating or updating distinct pull requests for each. Validates that branches are properly restacked before submitting, and fails if there are conflicts. Blocks force pushes to branches that overwrite branches that have changed since you last submitted or got them. Opens an interactive prompt that allows you to input pull request metadata. `gt ss` is a default alias for `gt submit --stack`.
#### 
[​](https://graphite.dev/docs/command-reference#flags-25)
flags
`--ai`Automatically AI-generate title and description for all PRs. Only works when creating new PRs. If —edit, use the generated metadata as starting points. `--always`Always push updates, even if the branch has not changed. Can be helpful for fixing an inconsistent Graphite stack view on Web/GitHub resulting from downtime/a bug. `--branch`Which branch to run this command from. Defaults to the current branch. `--cli`Edit PR metadata via the CLI instead of on web. `--comment`Add a comment on the PR with the given message. `-c, --confirm`Reports the PRs that would be submitted and asks for confirmation before pushing branches and opening/updating PRs. If either of —no-interactive or —dry-run is passed, this flag is ignored. `-d, --draft`If set, all new PRs will be created in draft mode. `--dry-run`Reports the PRs that would be submitted and terminates. No branches are restacked or pushed and no PRs are opened or updated. `-e, --edit`Input metadata for all PRs interactively. If neither —edit nor —no-edit is passed, only prompts for new PRs. `--edit-description`Input the PR description interactively. Default only prompts for new PRs. Takes precedence over —no-edit. `--edit-title`Input the PR title interactively. Default only prompts for new PRs. Takes precedence over —no-edit. `-f, --force`Force push: overwrites the remote branch with your local branch. Otherwise defaults to —force-with-lease. `--ignore-out-of-sync-trunk`Perform the submit operation even if the trunk branch is out of sync with its upstream branch. This can lead to incorrect metadata being used during the submit. `-m, --merge-when-ready`If set, marks all PRs being submitted as merge when ready, which will let them automatically merge as soon as all merge requirements are met. `--no-ai`Don’t use AI to generate any PR fields. Takes precedence over —ai. `-n, --no-edit`Don’t edit any PR fields inline. Takes precedence over —edit. `--no-edit-description`Don’t prompt for the PR description. Takes precedence over —edit-description and —edit. `--no-edit-title`Don’t prompt for the PR title. Takes precedence over —edit-title and —edit. `-p, --publish`If set, publishes all PRs being submitted. `--rerequest-review`Rerequest review from current reviewers. `--restack`Restack branches before submitting. If there are conflicts, output the branch names that could not be restacked `-r, --reviewers`If set without an argument, prompt to manually set reviewers. Alternatively, accepts a comma separated string of reviewers `-s, --stack`Submit descendants of the current branch in addition to its ancestors. `--target-trunk`Which trunk to open PRs against on remote. Defaults to the target trunk for the current local trunk (defined in `gt config`), or the current local trunk if no target trunk is configured. `-t, --team-reviewers`Comma separated list of team slugs. You can either pass “slug” to this flag or “org/slug” to the reviewers flag. Will enable the —reviewers prompt if set without arguments. `-u, --update-only`Only push branches and update PRs for branches that already have PRs open. `-v, --view`Open the PR in your browser after submitting. `-w, --web`Open a web browser to edit PR metadata, even if no new PRs are being created or if configured to edit PR metadata via the CLI.
### 
[​](https://graphite.dev/docs/command-reference#gt-sync)
`gt sync`
Sync all branches with remote, prompting to delete any branches for PRs that have been merged or closed. Restacks all branches in your repository that can be restacked without conflicts. If trunk cannot be fast-forwarded to match remote, overwrites trunk with the remote version.
#### 
[​](https://graphite.dev/docs/command-reference#flags-26)
flags
`-a, --all`Sync branches across all configured trunks. `-f, --force`Don’t prompt for confirmation before overwriting or deleting a branch in any place where confirmation is requested. `--restack`Restack any branches that can be restacked without conflicts (true by default; skip with —no-restack).
### 
[​](https://graphite.dev/docs/command-reference#gt-top)
`gt top`
Switch to the tip branch of the current stack. Prompts if ambiguous.
### 
[​](https://graphite.dev/docs/command-reference#gt-track-%5Bbranch%5D)
`gt track [branch]`
Start tracking the current (or provided) branch with Graphite by selecting its parent. Can recursively track a stack of branches by specifying each branch’s parent interactively. This command can also be used to fix corrupted Graphite metadata.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-17)
arguments
`[branch] (optional)`Branch to begin tracking. Defaults to the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-27)
flags
`-f, --force`Sets the parent to the most recent tracked ancestor of the branch being tracked to skip prompts. Takes precedence over —parent `-p, --parent`The tracked branch’s parent. Must be set to a tracked branch. If provided, only one branch can be tracked at a time.
### 
[​](https://graphite.dev/docs/command-reference#gt-trunk)
`gt trunk`
Show the trunk of the current branch.
#### 
[​](https://graphite.dev/docs/command-reference#flags-28)
flags
`--add`Add an additional trunk. `-a, --all`Show all configured trunks.
### 
[​](https://graphite.dev/docs/command-reference#gt-undo)
`gt undo`
Undo the most recent Graphite mutations.
#### 
[​](https://graphite.dev/docs/command-reference#flags-29)
flags
`-f, --force`Do not prompt for confirmation; undo the most recent command immediately.
### 
[​](https://graphite.dev/docs/command-reference#gt-unlink-%5Bbranch%5D)
`gt unlink [branch]`
Unlink the PR currently associated with the branch.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-18)
arguments
`[branch] (optional)`The branch to unlink.
### 
[​](https://graphite.dev/docs/command-reference#gt-untrack-%5Bbranch%5D)
`gt untrack [branch]`
Stop tracking a branch with Graphite. If the branch has children, they will also be untracked. Default to the current branch if none is passed in.
#### 
[​](https://graphite.dev/docs/command-reference#arguments-19)
arguments
`[branch] (optional)`Branch to stop tracking.
#### 
[​](https://graphite.dev/docs/command-reference#flags-30)
flags
`-f, --force`Will not prompt for confirmation before untracking a branch with children.
### 
[​](https://graphite.dev/docs/command-reference#gt-up-%5Bsteps%5D)
`gt up [steps]`
Switch to the child of the current branch. Prompts if ambiguous.
#### 
[​](https://graphite.dev/docs/command-reference#flags-31)
flags
`-n, --steps`The number of levels to traverse upstack.
Was this page helpful?
YesNo
[Previous](https://graphite.dev/docs/cli-quick-start)[ Command Cheatsheet Next ](https://graphite.dev/docs/cheatsheet)
On this page
  * [Global flags](https://graphite.dev/docs/command-reference#global-flags)
  * [Available commands](https://graphite.dev/docs/command-reference#available-commands)
  * [gt abort](https://graphite.dev/docs/command-reference#gt-abort)
  * [flags](https://graphite.dev/docs/command-reference#flags)
  * [gt absorb](https://graphite.dev/docs/command-reference#gt-absorb)
  * [flags](https://graphite.dev/docs/command-reference#flags-2)
  * [gt add [args..]](https://graphite.dev/docs/command-reference#gt-add-%5Bargs-%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments)
  * [gt aliases](https://graphite.dev/docs/command-reference#gt-aliases)
  * [flags](https://graphite.dev/docs/command-reference#flags-3)
  * [gt auth](https://graphite.dev/docs/command-reference#gt-auth)
  * [flags](https://graphite.dev/docs/command-reference#flags-4)
  * [gt bottom](https://graphite.dev/docs/command-reference#gt-bottom)
  * [gt changelog](https://graphite.dev/docs/command-reference#gt-changelog)
  * [gt checkout [branch]](https://graphite.dev/docs/command-reference#gt-checkout-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-2)
  * [flags](https://graphite.dev/docs/command-reference#flags-5)
  * [gt cherry-pick [args..]](https://graphite.dev/docs/command-reference#gt-cherry-pick-%5Bargs-%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-3)
  * [gt children](https://graphite.dev/docs/command-reference#gt-children)
  * [gt completion](https://graphite.dev/docs/command-reference#gt-completion)
  * [gt config](https://graphite.dev/docs/command-reference#gt-config)
  * [gt continue](https://graphite.dev/docs/command-reference#gt-continue)
  * [flags](https://graphite.dev/docs/command-reference#flags-6)
  * [gt create [name]](https://graphite.dev/docs/command-reference#gt-create-%5Bname%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-4)
  * [flags](https://graphite.dev/docs/command-reference#flags-7)
  * [gt dash](https://graphite.dev/docs/command-reference#gt-dash)
  * [gt delete [name]](https://graphite.dev/docs/command-reference#gt-delete-%5Bname%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-5)
  * [flags](https://graphite.dev/docs/command-reference#flags-8)
  * [gt demo [demoName]](https://graphite.dev/docs/command-reference#gt-demo-%5Bdemoname%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-6)
  * [gt docs](https://graphite.dev/docs/command-reference#gt-docs)
  * [gt down [steps]](https://graphite.dev/docs/command-reference#gt-down-%5Bsteps%5D)
  * [flags](https://graphite.dev/docs/command-reference#flags-9)
  * [gt feedback [message]](https://graphite.dev/docs/command-reference#gt-feedback-%5Bmessage%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-7)
  * [flags](https://graphite.dev/docs/command-reference#flags-10)
  * [gt fish](https://graphite.dev/docs/command-reference#gt-fish)
  * [gt fold](https://graphite.dev/docs/command-reference#gt-fold)
  * [flags](https://graphite.dev/docs/command-reference#flags-11)
  * [gt get [branch]](https://graphite.dev/docs/command-reference#gt-get-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-8)
  * [flags](https://graphite.dev/docs/command-reference#flags-12)
  * [gt guide [title]](https://graphite.dev/docs/command-reference#gt-guide-%5Btitle%5D)
  * [gt info [branch]](https://graphite.dev/docs/command-reference#gt-info-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-9)
  * [flags](https://graphite.dev/docs/command-reference#flags-13)
  * [gt init](https://graphite.dev/docs/command-reference#gt-init)
  * [flags](https://graphite.dev/docs/command-reference#flags-14)
  * [gt log [command]](https://graphite.dev/docs/command-reference#gt-log-%5Bcommand%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-10)
  * [flags](https://graphite.dev/docs/command-reference#flags-15)
  * [gt merge](https://graphite.dev/docs/command-reference#gt-merge)
  * [flags](https://graphite.dev/docs/command-reference#flags-16)
  * [gt modify](https://graphite.dev/docs/command-reference#gt-modify)
  * [flags](https://graphite.dev/docs/command-reference#flags-17)
  * [gt move](https://graphite.dev/docs/command-reference#gt-move)
  * [flags](https://graphite.dev/docs/command-reference#flags-18)
  * [gt parent](https://graphite.dev/docs/command-reference#gt-parent)
  * [gt pop](https://graphite.dev/docs/command-reference#gt-pop)
  * [gt pr [branch]](https://graphite.dev/docs/command-reference#gt-pr-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-11)
  * [flags](https://graphite.dev/docs/command-reference#flags-19)
  * [gt rebase [args..]](https://graphite.dev/docs/command-reference#gt-rebase-%5Bargs-%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-12)
  * [gt rename [name]](https://graphite.dev/docs/command-reference#gt-rename-%5Bname%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-13)
  * [flags](https://graphite.dev/docs/command-reference#flags-20)
  * [gt reorder](https://graphite.dev/docs/command-reference#gt-reorder)
  * [gt reset [args..]](https://graphite.dev/docs/command-reference#gt-reset-%5Bargs-%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-14)
  * [gt restack](https://graphite.dev/docs/command-reference#gt-restack)
  * [flags](https://graphite.dev/docs/command-reference#flags-21)
  * [gt restore [args..]](https://graphite.dev/docs/command-reference#gt-restore-%5Bargs-%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-15)
  * [gt revert [sha]](https://graphite.dev/docs/command-reference#gt-revert-%5Bsha%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-16)
  * [flags](https://graphite.dev/docs/command-reference#flags-22)
  * [gt split](https://graphite.dev/docs/command-reference#gt-split)
  * [flags](https://graphite.dev/docs/command-reference#flags-23)
  * [gt squash](https://graphite.dev/docs/command-reference#gt-squash)
  * [flags](https://graphite.dev/docs/command-reference#flags-24)
  * [gt submit](https://graphite.dev/docs/command-reference#gt-submit)
  * [flags](https://graphite.dev/docs/command-reference#flags-25)
  * [gt sync](https://graphite.dev/docs/command-reference#gt-sync)
  * [flags](https://graphite.dev/docs/command-reference#flags-26)
  * [gt top](https://graphite.dev/docs/command-reference#gt-top)
  * [gt track [branch]](https://graphite.dev/docs/command-reference#gt-track-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-17)
  * [flags](https://graphite.dev/docs/command-reference#flags-27)
  * [gt trunk](https://graphite.dev/docs/command-reference#gt-trunk)
  * [flags](https://graphite.dev/docs/command-reference#flags-28)
  * [gt undo](https://graphite.dev/docs/command-reference#gt-undo)
  * [flags](https://graphite.dev/docs/command-reference#flags-29)
  * [gt unlink [branch]](https://graphite.dev/docs/command-reference#gt-unlink-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-18)
  * [gt untrack [branch]](https://graphite.dev/docs/command-reference#gt-untrack-%5Bbranch%5D)
  * [arguments](https://graphite.dev/docs/command-reference#arguments-19)
  * [flags](https://graphite.dev/docs/command-reference#flags-30)
  * [gt up [steps]](https://graphite.dev/docs/command-reference#gt-up-%5Bsteps%5D)
  * [flags](https://graphite.dev/docs/command-reference#flags-31)


Assistant
Responses are generated using AI and may contain mistakes.
[Graphite home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-black.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/graphite-58cc94ce/logo/logo-white.svg)](https://graphite.dev)
[slack](https://community.graphite.dev)[x](https://twitter.com/withgraphite/)[github](https://github.com/withgraphite/)
Product
[Features](https://graphite.dev/features)[Pricing](https://graphite.dev/pricing)[Docs](https://graphite.dev/docs)[Customers](https://graphite.dev/customers)
Company
[Blog](https://graphite.dev/blog)[Careers](https://graphite.dev/careers)[Contact us](https://graphite.dev/contact-us)
Resources
[Community](https://community.graphite.dev)[Privacy policy](https://graphite.dev/privacy)[Terms of service](https://graphite.dev/terms-of-service)[Stacking workflow](https://graphite.dev/stacking)
Developers
[Status](https://status.graphite.dev)[GitHub](https://github.com/withgraphite/)
[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=graphite-58cc94ce)

