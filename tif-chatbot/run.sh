docker stop tif_be
docker rm tif_be
docker rmi sisai:tif_be

docker build -f dockerfile -t sisai:tif_be .
docker run -p 8080:8080 -d --name tif_be sisai:tif_be
docker logs -f tif_be