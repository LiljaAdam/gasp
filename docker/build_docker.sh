# Build the gasp Docker image.
# SSH agent is used for any private GitHub dependencies during build.

eval "$(ssh-agent -s)"

if [ -f ~/.ssh/id_ed25519 ]; then
    ssh-add ~/.ssh/id_ed25519
elif [ -f ~/.ssh/id_rsa ]; then
    ssh-add ~/.ssh/id_rsa
else
    echo "No recognized SSH key found in ~/.ssh"
fi

DOCKER_BUILDKIT=1 docker build --ssh default -t gasp:latest -f docker/Dockerfile --debug .
