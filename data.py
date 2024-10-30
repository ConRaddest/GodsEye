import requests

url = "http://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.json?api_key=7wWWn9VaOsY4eNGoLvdafSHnqZJu2n0R0IikztzM"

payload = "names=2018&leap_day=false&interval=15&utc=false&full_name=Connor%2BduToit&email=connordutoit@gmail.com&reason=Academic&wkt=MULTIPOINT(-106.22%2032.9741%2C-106.18%2032.9741%2C-106.1%2032.9741)"

headers = {
    'content-type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
}

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)