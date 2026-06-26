#### Initial Setup and Requirements

```bash
sudo apt install python3.13-venv # If you dont have it. Only needed once per system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Save dependencies

```bash
pip freeze > requirements.txt
```

#### How to run

```bash
source .venv/bin/activate
```