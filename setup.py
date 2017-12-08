from distutils.core import setup
setup(
  name = 'relaxflow',
  packages = ['relaxflow'], # this must be the same as the name above
  version = '0.1.0.1',
  description = 'A small library implementing the REBAR and RELAX stochastic gradient estimators in Tensorflow.',
  author = 'Rasmus Bonnevie',
  author_email = 'rasmusbonnevie@gmail.com',
  url = 'https://github.com/Bonnevie/rebar', # use the URL to the github repo
  download_url = 'https://github.com/Bonnevie/rebar/archive/0.1.0.1.tar.gz', # I'll explain this in a second
  keywords = ['rebar','relax','probability','expectation','estimator', 'gradient', 'tensorflow', 'discrete', 'sampling'], # arbitrary keywords
  classifiers = [],
)
