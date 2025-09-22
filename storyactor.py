import random
from mmap import ACCESS_COPY


class Actor:

    def __init__(self, name, actors):
        self.name = name
        self.trust = {}
        self.maxTrust = 20
        self.minTrust = 0
        self.aidReward = 2
        self.aidPenalty = -2
        for actor in actors:
            self.trust[actor] = (self.maxTrust + self.minTrust) / 2

    def set_actors(self, actors):
        for actor in actors:
            self.trust[actor] = (self.maxTrust + self.minTrust) / 2

    def request(self, actor):
        """  """
        if actor.aid(self):
            self.trust[actor] += self.aidReward
        else:
            self.trust[actor] += self.aidPenalty

    def aid(self, actor):
        """ Randomly chooses to aid based on actor's trust level """
        p = self.trust[actor] / (self.maxTrust-self.minTrust)
        return random.random() < p

    def __repr__(self):
        return self.name


class SelfishActor(Actor):

    def __init__(self, name, actors):
        super().__init__(name, actors)

    def aid(self, actor):
        """ Never chooses to aid """
        return False


class SelflessActor(Actor):

    def __init__(self, name, actors):
        super().__init__(name, actors)

    def aid(self, actor):
        """ Always chooses to aid """
        return

