1. Buiding docker ```docker build --file Dockerfile --tag fastapi-ml .```
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

