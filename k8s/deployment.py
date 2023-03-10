import logging
import yaml

from kubernetes import client
from kubernetes.client.rest import ApiException

from utils.probes import Probe
from k8s.pod import Pod
from k8s.client import get_apps_api
from exceptions import KubeException


logger = logging.getLogger(__name__)


class Deployment(object):

    def __init__(self, name=None, selector=None, labels=None, image_metadata=None,
                 namespace='default', create_in_cluster=False, from_template=None):
        """
        Utility functions for kubernetes deployments.

        :param name: str, name of the deployment
        :param selector: Label selector for pods. Existing ReplicaSets whose pods are selected by
         this will be the ones affected by this deployment. It must match the pod template's labels
        :param labels: dict, dict of labels
        :param image_metadata: ImageMetadata
        :param namespace: str, name of the namespace
        :param create_in_cluster: bool, if True deployment is created in Kubernetes cluster
        :param from_template: str, deployment template, example:
               - https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

        """

        self.namespace = namespace

        if (from_template is not None) and (name is not None or selector is not None or
                                            labels is not None or image_metadata is not None):
            raise KubeException(
                'from_template cannot be passed to constructor at the same time with'
                ' name, selector, labels or image_metadata')
        elif from_template is not None:
            self.body = yaml.safe_load(from_template)

            self.name = self.body['metadata']['name']

        elif (name is not None and selector is not None and
              labels is not None and image_metadata is not None):
            self.name = name
            self.pod = Pod.create(image_metadata)

            self.spec = client.V1DeploymentSpec(
                selector=client.V1LabelSelector(match_labels=selector),
                template=client.V1PodTemplateSpec(metadata=client.V1ObjectMeta(labels=selector),
                                                  spec=self.pod.spec))

            self.metadata = client.V1ObjectMeta(name=self.name, namespace=self.namespace,
                                                labels=labels)

            self.body = client.V1Deployment(spec=self.spec, metadata=self.metadata)
        else:
            raise KubeException(
                'to create deployment you need to specify template or'
                ' properties: name, selector, labels, image_metadata')

        self.api = get_apps_api()

        if create_in_cluster:
            self.create_in_cluster()

    def delete(self):
        """
        delete Deployment from the Kubernetes cluster
        :return: None
        """

        body = client.V1DeleteOptions()

        try:
            status = self.api.delete_namespaced_deployment(self.name, self.namespace, body)

            logger.info("Deleting Deployment %s in namespace: %s", self.name, self.namespace)
        except ApiException as e:
            raise KubeException(
                "Exception when calling Kubernetes API - delete_namespaced_deployment: %s\n" % e)

        if status.status == 'Failure':
            raise KubeException("Deletion of Deployment failed")

    def get_status(self):
        """
        get status of the Deployment
        :return: V1DeploymentStatus, https://git.io/vhKE3
        """
        try:
            api_response = self.api.read_namespaced_deployment_status(self.name, self.namespace)
        except ApiException as e:
            raise KubeException(
                "Exception when calling Kubernetes API - "
                "read_namespaced_deployment_status: %s\n" % e)

        return api_response.status

    def all_pods_ready(self):
        """
        Check if number of replicas with same selector is equals to number of ready replicas
        :return: bool
        """

        if self.get_status().replicas and self.get_status().ready_replicas:
            if self.get_status().replicas == self.get_status().ready_replicas:
                logger.info("All pods are ready for deployment %s in namespace: %s",
                            self.name, self.namespace)
                return True

        return False

    def wait(self, timeout=15):
        """
        block until all replicas are not ready, raises an exc ProbeTimeout if timeout is reached
        :param timeout: int or float (seconds), time to wait for pods to run
        :return: None
        """

        Probe(timeout=timeout, fnc=self.all_pods_ready, expected_retval=True).run()

    def create_in_cluster(self):
        """
        call Kubernetes API and create this Deployment in cluster,
        raise KubeException if the API call fails
        :return: None
        """
        try:
            self.api.create_namespaced_deployment(self.namespace, self.body)
        except ApiException as e:
            raise KubeException(
                "Exception when calling Kubernetes API - create_namespaced_deployment: %s\n" % e)

        logger.info("Creating Deployment %s in namespace: %s", self.name, self.namespace)