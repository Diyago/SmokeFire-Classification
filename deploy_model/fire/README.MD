Server ip 89.208.197.195

1. Buiding docker ```sudo docker build --file Dockerfile --tag fastapi-ml .```
For changing host change ip address in Dockerfile CMD ["main:app", "--host", "0.0.0.0"] to what ever you want

2. Wait up to 10 minutes //

3. Run docker ```sudo docker run -p 8000:8000 fastapi-ml```

4. Go to ```http://0.0.0.0:8000/docs``` for post get request API docs

5. Example POST request:

```curl -X POST "http://0.0.0.0:8000/predict_image/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test_image.png;type=image/png"```

6. Simple GET request:

```curl -X GET "http://0.0.0.0:8000/ping" -H "accept: application/json"```


7. Logs in ```file_1.log``` inside container
```
sudo docker exec -it 8b246bd73a5c bash
cat file_1.log
```


TODO use compose:

1. Install
```
sudo docker-compose up
```

2. Build
```
sudo docker-compose build
```

3. Run
```
sudo docker-compose up
```

4. Get to the docker
```
sudo docker exec -it  flaskhello_webapp_1 bash
```

5. Get active dockers
```
sudo docker ps
```

