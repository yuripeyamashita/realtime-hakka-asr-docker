FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y ffmpeg

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir gradio[oauth]==5.0.1 "uvicorn>=0.14.0" spaces
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["python", "app.py"]