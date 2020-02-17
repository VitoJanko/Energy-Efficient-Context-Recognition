Overview
========

Widespread accessibility of wearable sensing devices opens many possibilities for tracking the users who wear them. Possible applications range from measuring their exercise patterns and checking on their health, to determining their location. In this documentation we will use the term context-recognition for all these tasks.

A common problem when using such context-recognition systems is their impact on the battery life of the sensing device. It is easy to imagine that an application that monitors users' habits using all the sensors in a smartphone (accelerometer, GPS, Wi-Fi etc.) will quickly drain the phone's battery, making it useless in practice.

While many methods for reducing the energy-consumption of a context-recognition system already exist, most of them are specialized. They work either in a specific domain or can only optimize the energy consumption of specific sensors. Adapting these methods to another domain can be laborious and may require a lot of expert knowledge and experimentation.

We developed three novel methods for generating good energy-efficient solutions that are independent of the domain and can easily be used to optimize a wide range of context-recognition tasks. The first method,
""Setting-to-Context Assignment (SCA)"
changes the sensing settings -- which sensors to use, with what frequency, what duty cycles to use etc. -- depending on what context was last detected. To do so, we developed a mathematical model that can predict the performance of any given setting-to-context assignment. The SCA method then uses the NSGA-II algorithm to search the space of possible assignments, finding the best ones. The second method,
"Duty-Cycle-Assignment (DCA)"
works in a similar way to the SCA method, but is specialized in optimizing only duty-cycling, i.e., periodically turning the sensors on and off. %This specialization can be justified by the fact that duty-cycling is useful in basically all context-recognition tasks.

Finally, the "Cost-Sensitive Decision-Tree (CS-DT)"
method adapts sensing settings directly to the sensor data. It works by using a cost-sensitive decision tree that was adapted for context-recognition tasks. All three methods were then combined in three different ways in order to join their individual strengths. These combined methods can adapt to both the current context and to the sensor data.

Instead of returning only one solution, all of our methods can find different trade-offs between the classification quality and the energy consumption. Returning these trade-offs can help the system designer to pick one suitable for their system.

The methods are explained and in depth in PhD work <Insert a link>.
The methodology was also tested on four different real-life datasets, and on a family of artificial datasets. Doing so, we proved that it works under many different conditions, with different possible settings, sensors and problem domains. We also showed that our methods compare favourably against other state-of-the-art methods. For each of the tested datasets we found many energy-efficient solutions. For example, for the Commodity12 dataset, we reduced the energy consumption from 123 mA to 29 mA in exchange for less than 1 percentage point of accuracy. For another example, we were able to use only 5% of the available data in the SHL dataset (by using a lower frequency, duty-cycling and a subset of sensors) and in exchange sacrifice only 5% of the accuracy.
