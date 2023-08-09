# LEADVersion
This is the Mass-GT version implemented in the Digital Twin platform of the LEAD project. This version is based on a tactical freight simulator version of early 2021.

This repository consists on two versions of the MassGT implementation, each one in its own folder. The first folder called **"Developer version"** contains the version used for the development of the models and ran in local. The second folder called **"LEAD packaging"** has the version of the code implemented in docker containers in the Digital Twin platform of the LEAD project (and the implementation is tidier).

All models now export a KPI.json file with the main indicators of the run. 

## Structure of the model

The structure of this implementation is similar to the _master_ Mass-GT file where each submodule has its own individual module. In the LEAD project _connectors_ were implemented, that convert the outputs of each model into the inputs of the next. This was done in order to preserve as much as possible the modules that were included from the TFS (and make the use of the parcel market easier to implement).

### Parcel generation
Based on the parcel generator from the TFS (before the implementation of the household demand). It adds parcel locker demand and simulates Local-to-local  (L2L) demand. Adds a Crowdshipping eligibility column and classifies parcels between traditional hub-spoke (getting the treatment from TFS) and hyperconnected parcels (being subject to focused treatment according to the segment)

### Parcel Market
Generates individual networks per carrier and delivery method. The model allows for network sharing in the configuration file. With a shortest path each parcel trip is decomposed into individual trips. The shortest path is a weighted dijkstra algorith where a total cost can be estimated from time and cost. The network distinguishes between hub-hub trips and delivery to customer ones.

For crowdshipping trips, a disaggregated matching algorithm based on utlities is used to allocate the parcels to the travellers. 


### Parcel Tour formation
Using the same principles from the TFS, implments pick up tours for the L2L parcels at the end of the delivery tours. From the configuration file it allows different vehicle types between carriers and by the network they use (hub-hub or home delivery). The capacity and drop-off times of these vehicles is also edited in the configurated file.


### Parcel network
Virtually unchanged from TFS

### Connectors betweeen models
The connectors are implemented to transform the output of a model into the inputs of other. They mainly do some data wrangling and not actual modelling. 

#### Conn_Gen2Mkt
Converts the outputs from parcel generation to parcel market.
#### Conn_Mkt2Sched
Converts parcel market outputs into the inputs for parcel tour formation 
#### Conn_Sched2Ntwrk
Connects the parcel tour formation to the network module
#### Conn_Sched2COPERT
Changes the outputs from the parcel tour formation into COPERT. COPERT is a closed emission estimator used in LEAD for emission estimation
