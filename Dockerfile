FROM python:3.11

# ADD data_utils.py .
# ADD helper_utils.py .
# ADD models.py .

COPY . /rewheel

WORKDIR /code

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install /rewheel

# EXPOSE 8888

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["/bin/bash"]

# Make a non-root user
# Make cuda available

# TUTORIAL:
# Download MNIST data
# Install docker
# Install GPU extensions
# Start docker container
# Import MNIST data from local drive
# Setup Model
# Train model
# Export model to local drive
# Import model from local drive
# Export jupyter notebook from container to local drive
# Import jupyter notebook from local drive to container