# Intelligent Sonic the HedgeHog Agent

## Setup

1.) Clone the gym-retro repo:

    git clone --recursive https://github.com/openai/retro.git gym-retro
    cd gym-retro
    
2.) Import ROM and run a random agent

    python retro/scripts/import_sega_classics.py
    python examples/random_agent.py --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1
