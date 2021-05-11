# Intelligent Sonic the HedgeHog Agent
## Repository Overview
Each algorithm's code (NEAT, Q-Learning, A2C) is in its own branch.
## Setup

1.) Clone the gym-retro repo:

    git clone --recursive https://github.com/openai/retro.git gym-retro
    cd gym-retro
    
2.) Import ROM installed on computer. You will need to input your Steam credentials:

    python retro/scripts/import_sega_classics.py
    
3.) Run a random Sonic agent:
    
    python retro/examples/random_agent.py --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1
