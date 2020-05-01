# Audio captioning with language modelling

Translation metrics evaluated with https://github.com/tylin/coco-caption.git,
requires download of Stanford CoreNLP (with coco_caption/get_stanford_models.sh
or by following the commands there).

SPICE evaluation uses 8GB of RAM and METEOR uses 2GB (both use java). To change
RAM requirements go to coco_caption/pycocoevalcap and meteor/meteor.py:18 or
spice/spice.py:63 respectively and change the third argument of the java command.