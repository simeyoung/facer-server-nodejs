FROM node:8
WORKDIR /app
COPY package.json /app
ENV OPENCV4NODEJS_DISABLE_AUTOBUILD 1
RUN npm install
COPY . /app
CMD node index.js
EXPOSE 8082