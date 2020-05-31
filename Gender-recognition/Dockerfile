FROM continuumio/miniconda3
EXPOSE 5000
RUN conda create -n env python=3.7 pandas numpy matplotlib keras=2.2.4 scikit-learn flask
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
COPY . /app
WORKDIR /app
CMD python ./Main.py