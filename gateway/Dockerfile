# LLM gateway: agentgateway (https://agentgateway.dev) in front of Groq.
# The image's entrypoint is the agentgateway binary; the config is a Render
# Secret File (it contains the inbound API key), so nothing secret is in git.
#
# If the build says the tag doesn't exist, check the current tag on
# https://agentgateway.dev/docs/deployment/docker and change it here.
ARG AGENTGATEWAY_TAG=1.6.0
FROM cr.agentgateway.dev/agentgateway:${AGENTGATEWAY_TAG}

CMD ["-f", "/etc/secrets/config.yaml"]
