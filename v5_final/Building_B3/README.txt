================================================================================
  DISTRIBUTED STATE TRANSITION SYSTEM (DSTS)
  Building B3 -- Self-Contained Node
================================================================================

PREREQUISITES
-------------
  Python 3.8+
  pip install face_recognition numpy scikit-learn

SETUP
-----
  1. Open config.json in a text editor.
  2. Replace the IP addresses with everyone's real IPs.
     (If testing locally, leave them as 127.0.0.1)
  3. Make sure encodings_db.json exists in this folder.
     (It should already be included. If not, ask the person
      who set up the system to generate it.)

RUNNING
-------
  Option A -- Double-click run.bat (Windows only)

  Option B -- Two terminal windows:
    Terminal 1:  python building_server.py
    Terminal 2:  python simulate_cameras.py

QUERYING
--------
  Open a third terminal in this folder:

    python query_network.py --person B1_Person_2

  This will search ALL online buildings for that person's
  tracking history and save a detailed JSON report.

OCCUPANT IDS
------------
  Building B0: B0_Person_1 through B0_Person_5
  Building B1: B1_Person_1 through B1_Person_5
  Building B2: B2_Person_1 through B2_Person_5
  Building B3: B3_Person_1 through B3_Person_5
  Building B4: B4_Person_1 through B4_Person_5

FILES
-----
  building_server.py   -- TCP socket server (BSTS engine)
  simulate_cameras.py  -- Simulates 5 occupant camera feeds
  query_network.py     -- Search for any person across buildings
  config.json          -- Network topology (edit IPs here)
  encodings_db.json    -- LFW face encodings for all 25 occupants
  event_history.json   -- Created at runtime, logs all events
  search_result_*.json -- Created when you run a query
================================================================================
