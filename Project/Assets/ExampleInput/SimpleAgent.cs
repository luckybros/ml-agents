using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class SimpleAgent : Agent
{
    public Transform target;
    private Vector3 startPosition;

    public void Start()
    {
        startPosition = transform.localPosition;
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = startPosition;
    }

    // Non serve OnActionReceived per muovere l'oggetto!
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(target.localPosition);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            EndEpisode();
        }
    }
}
