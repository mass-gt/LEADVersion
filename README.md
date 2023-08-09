# LEADVersion
This is the Mass-GT version implemented in the Digital Twin platform of the LEAD project. This version is based on a tactical freight simulator version of early 2021.

This repository consists on two versions of the MassGT implementation, each one in its own folder. The first folder called "Developer version" contains the version used for the development of the models and ran in local. The second folder called "LEAD packaging" has the version of the code implemented in docker containers in the Digital Twin platform of the LEAD project.

## Structure of the model

The structure of this implementation is similar to the _master_ Mass-GT file where each submodule has its own individual module. In the LEAD project _connectors_ were implemented, that convert the outputs of each model into the inputs of the next. This was done in order to preserve as much as possible the modules that were included from the TFS (and make the use of the parcel market easier to implement).

### Parcel generation

### Parcel Market

### Parcel Tour formation

### Parcel network

### Connectors betweeen models

#### Conn_Gen2Mkt

#### Conn_Mkt2Sched

#### Conn_Sched2Ntwrk


#### Conn_Sched2COPERT

