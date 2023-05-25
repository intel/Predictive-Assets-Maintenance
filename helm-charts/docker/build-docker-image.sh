export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port


sudo docker build \
   --build-arg http_proxy=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT \
   --build-arg https_proxy=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT \
   -t intelanalytics/bigdl:asset-maintenance-ubuntu-20.04 .

