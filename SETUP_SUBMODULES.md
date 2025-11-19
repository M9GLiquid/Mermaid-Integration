# Setting Up Git Submodules

This document explains how to convert the API directories to Git submodules when you're ready to push them to remote repositories.

## Current Setup

The APIs are currently copied directly into `Integration-v1/apis/`:
- `apis/overlay-api/` - Overlay API
- `apis/layout-api/` - Layout API  
- `apis/hand-recognition-api/` - Hand Recognition API

Each API directory is already a Git repository (initialized locally).

## Converting to Git Submodules

### Step 1: Push API Repositories to Remote

For each API, push to a remote repository:

```bash
# For overlay-api
cd apis/overlay-api
git remote add origin <your-remote-url>/overlay-api.git
git push -u origin master

# For layout-api
cd ../layout-api
git remote add origin <your-remote-url>/layout-api.git
git push -u origin master

# For hand-recognition-api
cd ../hand-recognition-api
git remote add origin <your-remote-url>/hand-recognition-api.git
git push -u origin master
```

### Step 2: Remove Existing Directories

```bash
cd Integration-v1
rm -rf apis/overlay-api apis/layout-api apis/hand-recognition-api
```

### Step 3: Add as Git Submodules

```bash
git submodule add <your-remote-url>/overlay-api.git apis/overlay-api
git submodule add <your-remote-url>/layout-api.git apis/layout-api
git submodule add <your-remote-url>/hand-recognition-api.git apis/hand-recognition-api
```

### Step 4: Commit Submodule Configuration

```bash
git add .gitmodules apis/
git commit -m "Convert APIs to Git submodules"
```

## Using Submodules

### Initial Clone

When someone clones Integration-v1, they need to initialize submodules:

```bash
git clone <repository-url> Integration-v1
cd Integration-v1
git submodule update --init --recursive
```

### Updating APIs

To update all APIs to their latest versions:

```bash
git submodule update --remote
```

To update a specific API:

```bash
cd apis/overlay-api
git pull origin master
cd ../..
```

### Working with Submodules

- **Making changes to an API**: Work directly in the submodule directory, commit and push from there
- **Updating Integration-v1 to use new API version**: Update the submodule reference and commit

```bash
cd apis/overlay-api
git checkout <new-commit-or-branch>
cd ../..
git add apis/overlay-api
git commit -m "Update overlay-api to version X"
```

## Benefits

- **Version Control**: Each API has its own version history
- **Independent Updates**: APIs can be updated independently
- **Reusability**: APIs can be used in other projects
- **Clean Separation**: No copy-paste, APIs are referenced, not duplicated


