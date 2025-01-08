FROM node:20-alpine
ARG api_base_url

WORKDIR /app
COPY package*.json .
RUN npm i --silent
COPY . .
#RUN echo "VITE_API_BASE_URL = ${api_base_url}" > .env
RUN echo "VITE_API_BASE_URL = http://aitextdetector4-env.eba-fjz7gg3c.eu-north-1.elasticbeanstalk.com:8080" > .env
EXPOSE 5173
CMD ["npm", "run","dev"]
