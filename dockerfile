FROM runpod/base:0.6.2-cuda12.2.0

# Install dependencies
RUN apt-get update

WORKDIR /app

# Copy local source code instead of cloning
COPY . /app/ai-toolkit

WORKDIR /app/ai-toolkit

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt

RUN apt-get install -y tmux nvtop htop

COPY default /etc/nginx/sites-available/default
# Copy start.sh and make it executable
COPY start_train.sh /app/start_train.sh
RUN chmod +x /app/start_train.sh
COPY start_train.sh /app/start-no-files.sh
RUN chmod +x /app/start-no-files.sh
# Mask workspace
RUN mkdir /workspace

# Symlink app to workspace
RUN ln -s /app/ai-toolkit /workspace/ai-toolkit

# Expose necessary ports
EXPOSE 8888:8888  
EXPOSE 7860:7860  



#CMD ["/bin/bash", "/app/ai-toolkit/start_train.sh"]

