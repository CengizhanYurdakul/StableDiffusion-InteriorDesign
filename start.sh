sdName=$1
controlNetName=$2
dtype=$3
host=$4
port=$5
checkInput=$6


docker build -t interiordesign .
docker run --runtime=nvidia --shm-size="16g" -p $port:$port -it interiordesign --stableDiffusionModelName $sdName --controlnetMethod $controlNetName --dtype $dtype --host $host --port $port